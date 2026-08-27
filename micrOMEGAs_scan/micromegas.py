'''
The Omega(MAp, eps) oracle.

This is the only file that knows micrOMEGAs exists.  Everything above it --
the bracketing, the warm start, the walk -- sees a plain function of two
numbers, which is what lets the same driver run against the mock oracle at
the bottom of this file.

A call writes the four model parameters to the .par file, runs the executable
and parses its output.  Anything that goes wrong -- a nonzero exit, a
timeout, output that does not parse -- comes back as nan rather than an
exception, so a bad point is a failed point and not a dead scan.
'''

import math
import os
import subprocess
import threading

import config


# Columns of the evaluation cache.  alpha_D and the mass ratio lead, because
# they are what makes a cached Omega belong to one run rather than another.
CACHE_HEADER = 'alphaD,mchi_over_map,MAp,eps,omegah2,xf'


def _fmt(x):
    '''Value formatting for the .par file: enough digits to round-trip.'''
    return f'{float(x):.12g}'


def parse_output(text):
    '''Pull (xf, omegah2) out of the executable's stdout.

    The expected output is a single line of two numbers, "1.43e+01 2.67e+02",
    xf first and Omega h^2 second.  Banner lines and warnings are tolerated by
    scanning from the bottom for the last line that is exactly two floats.
    '''
    for line in reversed(text.strip().splitlines()):
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            xf, om = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        return xf, om
    return math.nan, math.nan


class Oracle:
    '''Omega h^2 as a function of (MAp, eps), with alpha_D and the mass ratio fixed.

    Calling the instance returns Omega alone, which is what the root finders
    want.  xf comes back from lookup() and is carried through to the output,
    since micrOMEGAs hands it over for free and it is the quantity that makes
    d log Omega / d log eps drift away from -2.

    Every evaluation is cached in memory and appended to a CSV, so a scan that
    is interrupted and restarted with the same grid replays its earlier work
    without spawning a single process.  The scan is deterministic, so the
    replay is exact.
    '''

    def __init__(self, micromegas_path=None, executable=None, par_file_name=None,
                 alpha_d=None, mchi_over_map=None, gd=None, timeout=None,
                 cache_path=None, verbose=False):
        self.path = (micromegas_path or config.MICROMEGAS_PATH)
        self.executable = executable or config.EXECUTABLE
        self.par_file = self.path / (par_file_name or config.PAR_FILE_NAME)
        self.mchi_over_map = (config.MCHI_OVER_MAP if mchi_over_map is None
                              else mchi_over_map)
        if gd is not None:
            self.gd = float(gd)
            self.alpha_d = self.gd**2/(4.0*math.pi)
        elif alpha_d is not None:
            self.alpha_d = float(alpha_d)
            self.gd = math.sqrt(4.0*math.pi*self.alpha_d)
        else:
            self.gd = config.gD_value()
            self.alpha_d = self.gd**2/(4.0*math.pi)
        self.timeout = config.TIMEOUT if timeout is None else timeout
        self.verbose = verbose

        self.ncalls = 0          # processes actually spawned
        self.nlookups = 0        # requests, cache hits included
        self.nfailed = 0
        self._cache = {}
        self.cache_path = cache_path
        if cache_path is not None:
            self._load_cache()
            self._cache_fh = open(cache_path, 'a', buffering=1)
            if self._cache_fh.tell() == 0:
                self._cache_fh.write(CACHE_HEADER + '\n')
        else:
            self._cache_fh = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------- cache

    def _key(self, ma, eps):
        # alpha_D and the mass ratio are part of the identity of a point, not
        # just MAp and eps.  They are fixed within a run but not between runs,
        # and a key without them hands back the previous alpha_D's Omega after
        # the config is edited -- silently, since a cache hit spawns no process
        # and so does not even rewrite the .par file.
        #
        # 12 significant digits: finer than any step the root finders take,
        # coarse enough that a replayed scan lands on the same key.
        return (f'{self.alpha_d:.12e}', f'{self.mchi_over_map:.12e}',
                f'{float(ma):.12e}', f'{float(eps):.12e}')

    def _load_cache(self):
        try:
            fh = open(self.cache_path)
        except OSError:
            return
        with fh:
            header = fh.readline().strip()
            if header != CACHE_HEADER:
                raise ValueError(
                    f'{self.cache_path} is not a cache file in the current '
                    f'format.\n  expected header: {CACHE_HEADER}\n  '
                    f'found:           {header!r}\n'
                    f'Caches written before alpha_D and Mchi/MAp were part of '
                    f'the key cannot be read, because nothing in them says '
                    f'which alpha_D they were computed at.  Delete the file or '
                    f'point --cache somewhere else.')
            for line in fh:
                parts = line.strip().split(',')
                if len(parts) != 6:
                    continue
                try:
                    ad, ratio, ma, eps, om, xf = (float(p) for p in parts)
                except ValueError:
                    continue            # a torn line
                # Rows for other alpha_D or mass ratios keep their own keys and
                # simply never match, so one file can hold several scans.
                self._cache[(f'{ad:.12e}', f'{ratio:.12e}',
                             f'{ma:.12e}', f'{eps:.12e}')] = (om, xf)

    def _store(self, ma, eps, om, xf):
        self._cache[self._key(ma, eps)] = (om, xf)
        if self._cache_fh is not None:
            self._cache_fh.write(f'{self.alpha_d:.12e},{self.mchi_over_map:.12e},'
                                 f'{ma:.12e},{eps:.12e},{om:.12e},{xf:.12e}\n')

    # ------------------------------------------------------------- calls

    def _run(self, ma, eps):
        mchi = self.mchi_over_map*ma
        names = config.PAR_NAMES
        card = (f"{names['mchi']} {_fmt(mchi)}\n"
                f"{names['map']} {_fmt(ma)}\n"
                f"{names['eps']} {_fmt(eps)}\n"
                f"{names['gD']} {_fmt(self.gd)}\n")
        with open(self.par_file, 'w') as fh:
            fh.write(card)

        try:
            result = subprocess.run([f'./{self.executable}', str(self.par_file)],
                                    cwd=self.path, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True,
                                    timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self._warn(ma, eps, f'timed out after {self.timeout:g} s')
            return math.nan, math.nan
        except OSError as exc:
            self._warn(ma, eps, f'could not run {self.executable}: {exc}')
            return math.nan, math.nan

        if result.returncode != 0:
            self._warn(ma, eps, f'exit code {result.returncode}: '
                                f'{result.stderr.strip()[:200]}')
            return math.nan, math.nan

        xf, om = parse_output(result.stdout)
        if not math.isfinite(om):
            self._warn(ma, eps, f'unparseable output: {result.stdout.strip()[:200]!r}')
            return math.nan, math.nan
        if om <= 0:
            # micrOMEGAs does return negative Omega -- darkOmega failing to
            # integrate rather than a physical answer.  Treated as a failed
            # point: the root finders need Omega > 0 to work in logs.
            self._warn(ma, eps, f'micrOMEGAs returned Omega h^2 = {om:.3g} <= 0'
                                f' (xf = {xf:.3g})')
            return math.nan, math.nan
        return om, xf

    def _warn(self, ma, eps, msg):
        self.nfailed += 1
        print(f'  ! MAp = {ma:.6g}, eps = {eps:.6g}: {msg}')

    def lookup(self, ma, eps):
        '''(Omega h^2, xf) at this point, from cache or from micrOMEGAs.'''
        self.nlookups += 1
        key = self._key(ma, eps)
        if key in self._cache:
            return self._cache[key]
        with self._lock:
            self.ncalls += 1
            om, xf = self._run(ma, eps)
        self._store(ma, eps, om, xf)
        if self.verbose and math.isfinite(om):
            print(f'    MAp = {ma:.6g}  eps = {eps:.6g}  ->  '
                  f'Omega h^2 = {om:.6g}  (xf = {xf:.4g})')
        return om, xf

    def __call__(self, ma, eps):
        return self.lookup(ma, eps)[0]

    def close(self):
        if self._cache_fh is not None:
            self._cache_fh.close()
            self._cache_fh = None

    def discard_cache(self):
        '''Close the cache and delete the file.

        The cache exists so an interrupted scan can resume; once a scan has
        finished there is nothing left to resume, and what remains is a file
        that goes stale without any way to notice. Its keys record alpha_D and
        the mass ratio, but nothing records the model, so recompiling
        micrOMEGAs or swapping work/models leaves every entry wrong and every
        key still matching. Deleting it on success keeps the resume and drops
        the hazard.

        Returns the path removed, or None if there was no cache file.
        '''
        self.close()
        if self.cache_path is None:
            return None
        try:
            os.remove(self.cache_path)
        except OSError:
            return None
        return self.cache_path


class MockOracle:
    '''Analytic stand-in for micrOMEGAs, for exercising the scan without it.

    Omega is built to look like the real thing where it matters to the solver:
    it falls off as a power of eps close to -2 but not equal to it, and the
    eps that hits the target rises with the mass, so the extrapolation and the
    secant slope are both tested.  It is not physics.
    '''

    def __init__(self, target=0.12, slope=-1.85, eps_ref=3e-4, ma_ref=0.05,
                 mass_power=0.75, **_ignored):
        self.target, self.slope = target, slope
        self.eps_ref, self.ma_ref, self.mass_power = eps_ref, ma_ref, mass_power
        self.ncalls = self.nlookups = self.nfailed = 0
        self._cache = {}
        self.cache_path = None

    def eps_true(self, ma):
        '''The relic line this mock defines, for checking what the scan found.'''
        return self.eps_ref*(ma/self.ma_ref)**self.mass_power

    def lookup(self, ma, eps):
        self.ncalls += 1
        self.nlookups += 1
        om = self.target*(eps/self.eps_true(ma))**self.slope
        xf = 20.0 + 0.8*math.log(max(eps, 1e-300)/self.eps_true(ma))
        self._cache[(ma, eps)] = (om, xf)
        return om, xf

    def __call__(self, ma, eps):
        return self.lookup(ma, eps)[0]

    def close(self):
        pass

    def discard_cache(self):
        return None
