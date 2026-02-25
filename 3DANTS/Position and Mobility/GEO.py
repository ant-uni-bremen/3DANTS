# satellites modelling parameters calculator 

import numpy as np
import datetime
from sgp4.api import Satrec, WGS72
import skyfield.api as sf
from sympy import symbols, solve
from progress.bar import Bar
import pandas as pd
import math

class LEO_GEO():

    def __init__(self, r_E, gm, h_GEO):
           self.r_E = r_E   # Earth radius
           self.gm = gm    # Geocentric Gravitational Constant [m^3 s^−2]
           self.h_GEO = h_GEO # The altitude of GEO satellites in meter

    def motion(self, height, x):
        v1 = np.sqrt(self.gm / (height))    # speed in meter / second
        if x == 'meter/sec':
           v = v1    # speed in meter / second
        elif x == 'rad/min':
             v = (v1 / (height)) * 60  # speed in radians / minutes
        return v

    def GEO(self, inc_GEO):
        r_GEO = self.h_GEO + self.r_E    # The radius of GEO satellite in meter
        incRad = inc_GEO * np.pi / 180    # The inclination angle of orbit
        v_GEO = self.motion(r_GEO,'meter/sec')    # speed in meter / second
        v_GEO_rad_min = self.motion(r_GEO,'rad/min')  # speed in radians / minutes
        print('v_LEO_rad_min'+str(v_GEO_rad_min))
        period_time_second = (2*np.pi*r_GEO)/(v_GEO)    #Period time in seconds
        period_time_minutes = period_time_second / 60
        raan = 0
        ma = 0
        days = self.Epoch_time()
        satrec = Satrec()
        satrec.sgp4init(
                WGS72,  # gravity model
                'i',    # keep as is, 'i' = improved mode
                1,      # satnum: Satellite number
                days,   # epoch: days since 1949 December 31 00:00 UT
                0,      # bstar: drag coefficient (idealized: 0)
                0,      # ndot: ballistic coefficient (revs/day), 0 is idealized
                0,      # nddot: mean motion 2nd derivative (revs/day^3), 0 is idealized
                0,      # ecco: eccentricity
                0,      # argpo: argument of perigee (radians)
                incRad, # inclo: inclination (radians)
                ma,     # mo: mean anomaly (radians)
                v_GEO_rad_min,  # no_kozai: mean motion (radians/minute)
                raan    # nodeo: right ascension of ascending node (radians)
            ) # https://rhodesmill.org/skyfield/earth-satellites.html
        #print('satrec = ' + str(satrec))
        ts = sf.load.timescale()    #Create time scale
        #print('ts'+str(ts))
        GEO_sat = sf.EarthSatellite.from_satrec(satrec, ts)

        #t = ts.utc(2021, 6, 29, 00, 00, 00)
        #geocentric = sat.at(t)
        #print(geocentric.position.km)
        return GEO_sat

    def Epoch_time(self):
        # SGP4 epoch is days since 1949-12-31 00:00 UTC.
        # Use current UTC time so freshly-created satellites have a valid,
        # up-to-date epoch and do not accumulate propagation error from
        # a stale hard-coded date.
        epoch = datetime.datetime(1949, 12, 31, 0, 0, 0)
        now   = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        diff  = now - epoch
        days  = diff.days + (diff.seconds / (24 * 3600))
        return days
    #UTC — Coordinated Universal Time (“Greenwich Time”)
    #UT1 — Universal Time
    #TAI — International Atomic Time
    #TT — Terrestrial Time
    #TDB — Barycentric Dynamical Time (the JPL’s Teph)

    def difference_time_in_seconds(self, t1, t2):
       #t1 and t2 in this format t1 = ts.utc(2021,7,22,10,10,12) , t2 = ts.utc(2022,7,23,20,10,9)
       difference_time_in_seconds = round((t2.toordinal() - t1.toordinal())*(24*60*60))
       # The following approach also results in the above approach
       '''
       time1 = "2021-07-22 10:10:09"
       t1 = datetime.datetime.fromisoformat(time1)

       time2 = "2021-07-23 09:10:09"
       t2 = datetime.datetime.fromisoformat(time2)

       difference_time = t2 - t1
       difference_time_in_seconds = (difference_time.days)*24*60*60 + (difference_time.seconds)
       '''
       return difference_time_in_seconds

    def walkerConstellation(self, height, inclination, numSat, numPlanes, phasing, name = "Sat"):
     """Walker Delta Pattern Constellation i: t/p/f

     i: inclination [deg]
     t: total number of satellites
     p: number of orbital planes (equally spaced)
     f: relative spacing between satellites in adjacent planes (0 .. p-1)

     height: oribtal height above Earth [m]
     inclincation: i [deg]
     numSat: t
     numPlanes: p
     phasing: f

     ts : skyfield timescale
     rE: Earth radius [m]
     gm: Geocentric Gravitational Constant [m^3 s^−2]
     """

     ts = sf.load.timescale()

     # check number of satellites
     S = numSat / numPlanes
     assert S == int(S), "numSat / numPlanes is not integer"
     S = int(S)

     # convert parameters
     r_LEO = self.r_E + height # orbital radius
     incRad = inclination * np.pi / 180
     #motion = np.sqrt(self.gm / (ro)**3) * 60 # speed in radians / minute
     v_LEO = self.motion(r_LEO,'meter/sec')    # speed in meter / second
     v_LEO_rad_min = self.motion(r_LEO,'rad/min')  # speed in radians / minutes
     print('v_LEO_rad_min'+str(v_LEO_rad_min))
     period_time_second = (2*np.pi*r_LEO)/(v_LEO)    #Period time in seconds
     period_time_minutes = period_time_second / 60   #Period time in minutes


     # get epoch (now)
     days = self.Epoch_time()

     # build constellation
     satellites = list()
     cnt = 0

     for i in range(numPlanes):
        print('numPlanes = ' + str(numPlanes))
        # formulas raan and ma taken from doi:10.3390/rs12111845

        # right ascension of the ascending nodes (RAAN)
        raan = i / numPlanes * 2 * np.pi

        for j in range(S):
            cnt = cnt + 1
            # mean anomaly
            ma = 2 * (j / S + phasing * i / numSat) * np.pi

            satrec = Satrec()
            satrec.sgp4init(
                    WGS72,  # gravity model
                    'i',    # keep as is
                    cnt,      # increment per satellite
                    days,   # epoch (keep)
                    0,      # drag coefficient (idealized: 0)
                    0,      # not used, 0 is idealized
                    0,      # not used, 0 is idealized
                    0,      # eccentricity
                    0,      # argument of perigee (radians)
                    incRad,
                    ma,      # mean anomaly
                    v_LEO_rad_min, # mean motion
                    raan   # right ascension of ascending node
                ) # https://rhodesmill.org/skyfield/earth-satellites.html

            sat = sf.EarthSatellite.from_satrec(satrec, ts)
            sat.name = "{} {}".format(name, cnt)
            satellites.append(sat)
        print('satellites = ' + str(satellites))
     return satellites


    def simulateConstellation(self, satellites, groundstation, minimumElevation, startTime, stopTime, ts = None, safetyMargin = 0):
        """Simulate visibility from <groundstation> for satellites in constellation <satellites> at a minimum elevation angle <minimumElevation> from <startTime> to <stopTime>

    Optional arguments:
        ts : skyfield timescale
        safetyMargin: simulate <safetyMargin> days before and after time span. This is necessary because otherwise there might be set/rise missing from dataframe

    Output:
        pandas.DataFrame, indexed with satellite names. Contains the columns:
            Rise: time the satellite rises above minimum elevation angle, as seen from groundstation
            Set: time the satellite sets below the minimum elevation angle, as seen from groundstation
            Visibility: The time between Rise and Set
            Offline: The time between Set and the next Rise of that satellite
        """
        # init time scale
        if ts is None:
            ts = sf.load.timescale()

        # load constellation
        #print('Loaded', len(satellites), 'satellites for starlink constellation with epoch', satellites[0].epoch.utc_jpl().removeprefix('A.D. '))

        # time span
        tspan = (startTime, stopTime)
    #    print('Simulating from {} until {}'.format(*[s.utc_jpl().removeprefix('A.D. ') for s in tspan]))

        real_tspan = (ts.tt_jd(tspan[0].tt-safetyMargin), ts.tt_jd(tspan[1].tt+safetyMargin)) # safety margin
        #print('real_tspan'+str(real_tspan))

        # sat visibility finder
        def satVisibility(sat):
            t, events = sat.find_events(groundstation, real_tspan[0], real_tspan[1], altitude_degrees = minimumElevation)
            print('events' + str(events))
            sat_rise = list()
            sat_set = list()
            lastevent = -1
            for ti, event, cnt in zip(t, events, range(len(events))):
                if lastevent == -1:
                    if event != 0:
                        continue
                    else:
                        lastevent = 2

                if event == 0: # rise
                    if lastevent != 2:
                        if sat_rise[-1] == pd.Timestamp(ti.utc_iso()):
                            # work around bug in starfield
                            pass
                        else:
                            raise RuntimeError("satellite did not set")
                    else:
                        lastevent = 0
                        sat_rise.append(pd.Timestamp(ti.utc_iso()))

                elif event == 1: # culminate
                    if lastevent != 0 and lastevent != 1:
                        if abs((sat - groundstation).at(ti).altaz()[0].degrees - minimumElevation) <= 1:
                            # work around bug in starfield
                            sat_rise.append(pd.Timestamp(ti.utc_iso()))
                        elif events[cnt+1] == 0 and ti == t[cnt+1]:
                            # work around bug in starfield
                            sat_rise.append(pd.Timestamp(ti.utc_iso()))
                        else:
                            raise RuntimeError("satellite did not rise")

                    lastevent = 1


                elif event == 2: # set
                    if lastevent != 1:
                        raise RuntimeError("satellite did not culminate")
                    lastevent = 2

                    sat_set.append(pd.Timestamp(ti.utc_iso()))

                else:
                    raise RuntimeError("unknown event")

            if event != 2:
                sat_rise.pop()

            df = pd.DataFrame({'Rise': sat_rise, 'Set': sat_set, 'Satellite': sat.name})

            # calculate offline times
            t = df['Rise'].copy()
            t.index = t.index-1
            t = t.drop(index=-1)
            df.insert(len(df.columns), 'Offline', t-df['Set'])
            df.insert(len(df.columns), 'end_time', df['Set'])

            # bracketing
            idx = np.logical_and(df['Set'] >= pd.Timestamp(tspan[0].utc_iso()), df['Rise'] <= pd.Timestamp(tspan[1].utc_iso()))

            # degenerate visits
            idx = np.logical_and(idx, df['Set'] != df['Rise'])

            # drop
            df = df[idx].reset_index().drop(columns='index')

            assert(np.all(df['Rise'] < df['Set']))
            return df

        # find satellite rise and set
        df = None
        with Bar(' ...', max = len(satellites), suffix = '%(index)d/%(max)d  ETA: %(eta)g s') as bar:
            for sat in satellites:
                if df is None:
                    df = satVisibility(sat)
                else:
                    df = pd.concat([df, satVisibility(sat)])

                bar.next()

        # post process
        df.sort_values(by=['Rise','Set'], inplace=True)
        df.insert(2, 'Visibility', df['Set']-df['Rise'])
        df = df.set_index('Satellite')

        return df

    def distance(self, P_GEO, P_LEO):   # In km
        distance = np.sqrt((( P_GEO[0] - P_LEO[0] )**2) + (( P_GEO[1] - P_LEO[1] )**2) + (( P_GEO[2] - P_LEO[2] )**2))
        return distance


    def GEO_LEO_max_distance(self, h_GEO, h_LEO):   # The output is km
        r_GEO = h_GEO + self.r_E    #The distance of GEO satellite to the Earth center in km
        r_LEO = h_LEO + self.r_E    #The distance of LEO satellite to the Earth center in km
        max_distance = np.sqrt( r_GEO**2 - self.r_E**2 ) + np.sqrt( r_LEO**2 - self.r_E**2 ) #In meters
        max_distance = max_distance / 1000  #In km
        return max_distance #In km
    
    
    def elevation_angel_calculator(self, P_sat, P_gs):
        sat_pos_ecef = P_sat
        obs_pos_ecef = P_gs
        # Calculate vector from observer to satellite
        r_obs_sat = sat_pos_ecef - obs_pos_ecef
        # Convert vector to local-level coordinates:
         # Define WGS84 ellipsoid constants
        a = 6378137.0  # semimajor axis (m)
        b = 6356752.314245  # semiminor axis (m)
        f = (a - b) / a  # flattening
        e_sq = f * (2 - f)  # eccentricity squared
        # Calculate magnitude of position vector
        r = np.linalg.norm(obs_pos_ecef)
        # Calculate latitude
        obs_lat = math.atan2(obs_pos_ecef[2], math.sqrt(obs_pos_ecef[0]**2 + obs_pos_ecef[1]**2))
        # Calculate longitude
        obs_lon = math.atan2(obs_pos_ecef[1], obs_pos_ecef[0])
        # Calculate altitude
        N = a / math.sqrt(1 - e_sq * math.sin(obs_lat)**2)
        alt = r - N
        # Convert latitude and longitude to degrees
        # No manual correction: geodetic latitude is already correctly derived
        # from the ECEF position vector above.
        obs_lat = math.radians(math.degrees(obs_lat))
        obs_lon = math.radians(math.degrees(obs_lon))
        #obs_lat = math.radians(53.105750)  # observer's latitude in radians
        #obs_lon = math.radians(8.859860)  # observer's longitude in radians
        obs_n = -np.sin(obs_lat) * np.cos(obs_lon) * r_obs_sat[0] - np.sin(obs_lat) * np.sin(obs_lon) * r_obs_sat[1] + np.cos(obs_lat) * r_obs_sat[2]
        obs_e = -np.sin(obs_lon) * r_obs_sat[0] + np.cos(obs_lon) * r_obs_sat[1]
        obs_u = np.cos(obs_lat) * np.cos(obs_lon) * r_obs_sat[0] + np.cos(obs_lat) * np.sin(obs_lon) * r_obs_sat[1] + np.sin(obs_lat) * r_obs_sat[2]
        # Calculate elevation angle in degrees
        elevation_angle = math.degrees(math.atan2(obs_u, math.sqrt(obs_e**2 + obs_n**2)))
        # Return the true computed elevation angle.
        # Callers should apply their own minimum-elevation mask if needed
        # (e.g. simulateConstellation already filters by minimumElevation).
        return elevation_angle

'''
    def GEO_LEO_max_distance_outer_product_appeoach(self, P_GEO, P_LEO):
        max_distance = (np.linalg.norm(P_GEO)*np.linalg.norm(P_LEO))/(np.linalg.norm( P_GEO - P_LEO ))

        return max_distance

    def GEO_LEO_max_distance_radius_approach(self, h_GEO, h_LEO, P_GEO, P_LEO):
        r_GEO = h_GEO + self.r_E
        r_LEO = h_LEO + self.r_E
        r_GEO = r_GEO /1000
        r_LEO = r_LEO /1000
        dist = (self.distance(P_GEO, P_LEO))
        print('r_GEO' + str(r_GEO))
        print('dist' + str(dist))
        x = symbols('x')
        expr = ((r_LEO**2 - x**2)**(1/2)) + ((r_GEO**2 - x**2)**(1/2)) - (dist)
        max_distance = solve(expr)
        print('max_distance first = '+str(max_distance))
        if max_distance != []:
          max_distance = max(max_distance)

        print('max_distance = '+str(max_distance))
        #max_distance = max_distance / 1000
        return max_distance
'''
