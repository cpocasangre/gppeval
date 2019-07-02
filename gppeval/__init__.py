# -*- coding: utf-8 -*-
"""
Title:
                    gppeval (geothermal power potential evaluation)
Description:
                    A Python-based stochastic library for assessing geothermal power potential
                    using the volumetric method in a liquid-dominated reservoir.
Author:
                    Carlos O. Pocasangre Jimenez
Supervisor:
                    Yasuhiro Fujimitsu
Organization:
                    1. Department of Earth Resources Engineering, Kyushu University.
                    744 Motooka, Nishi-ku, Fukuoka 819-0395, Japan

                    2. School of Electrical Engineering, University of El Salvador.
                    At the end of North 25th Avenue "Mártires 30 de julio", San Salvador
Date:
                    Created on Mon Dec 12th 2017
Last modification:
                    ( ... Wed Apr 17th 2019)
Version:
                    2019.4.17.0.3.dev1
Python_version:
                    3.5
Abstract:
                    We present a Python-based stochastic library for assessing geothermal power
                    potential using the volumetric method in a liquid-dominated reservoir.
                    The specific aims of this study are to use the volumetric method, “heat in
                    place,” to estimate electrical energy production ability from a geothermal
                    liquid-dominated reservoir, and to build a Python-based stochastic library
                    with useful methods for running such simulations. Although licensed
                    software is available, we selected the open-source programming language
                    Python for this task. The Geothermal Power Potential Evaluation stochastic
                    library (gppeval) is structured as three essential objects including a
                    geothermal power plant module, a Monte Carlo simulation module, and a tools
                    module. In this study, we use hot spring data from the municipality of
                    Nombre de Jesus, El Salvador, to demonstrate how the gppeval can be used to
                    assess geothermal power potential. Frequency distribution results from the
                    stochastic simulation shows that this area could initially support a
                    9.16-MWe power plant for 25 years, with a possible expansion to 17.1 MWe.
                    Further investigations into the geothermal power potential will be
                    conducted to validate the new data.
Contact:
                    carlos.pocasangre@mine.kyushu-u.ac.jp

                    carlos.pocasangre@fia.ues.edu.sv
Reference:
                    Pocasangre, C., & Fujimitsu, Y. (2018). A Python-based stochastic library
                    for assessing geothermal power potential using the volumetric method in a
                    liquid-dominated reservoir. Geothermics, 76, 164–176

                    https://doi.org/10.1016/J.GEOTHERMICS.2018.07.009
"""


import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import mcerp as mc
import numpy as np
from time import clock
import scipy.stats as ss
from beautifultable import BeautifulTable

__version_info__ = (2019, 4, 17, 0, 3, 'dev1')
__version__ = '.'.join(map(str, __version_info__))
__author__ = 'Carlos O. POCASANGRE JIMENEZ'
__description__ = 'Geothermal Power Potential assessment'
__url__ = 'https://github.com/cpocasangre/gppeval'
__module_name__ = 'gppeval'
__author_email__ = 'carlos.pocasangre@mine.kyushu-u.ac.jp'
__license__ = 'MIT License'
__status__ = 'Development release'


class Reservoir(object):
    """
    **Reservoir abstraction**

    usage:
        reservoir_instance = gppeval.Reservoir()

    :param name: name of the reservoir
    :param location: dictionary with coordinates of center point in degree
    :param address: address of reservoir
    :param area: reservoir geometric surface area [km2]
    :param thickness: reservoir thickness [m]
    :param volume: volume [km3]
    """
    def __init__(self, **kwargs):
        self.name = 'Default name'
        self.location = {'lat': 0.0, 'lon': 0.0}
        self.address = 'Default address'

        # by default, area, thickness, and volume are set to pdf = None. If the user wants to
        # use either V = A * h or V = V, it must be defined before.

        self.area = {'min': 1.0, 'most_likely': 1.0, 'max': 1.0, 'mean': 1.0, 'sd': 1.0,
                     'pdf': None}
        self.thickness = {'min': 1.0, 'most_likely': 1.0, 'max': 1.0, 'mean': 1.0,
                          'sd': 1.0, 'pdf': None}
        self.volume = {'min': 1.0, 'most_likely': 1.0, 'max': 1.0, 'mean': 1.0,
                          'sd': 1.0, 'pdf': None}
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
        super(Reservoir, self).__init__()

    @staticmethod
    def set_values_to_variables(var, values):
        """
        helping function for giving values to variables which accept kwargs

        :param var: variable to change dictionary
        :param values: dictionary with variable values
        """
        for key in values.keys():
            if key in var:
                var[key] = values[key]

    def get_name(self):
        """
        Get name

        usage:
            string = get_name()

        :return name: name of the reservoir (string)
        """
        return self.name

    def get_location(self):
        """
        Get location dictionary, latitude and longitude

        usage:
            dictionary = get_location()

        :return location: a dictionary with latitude and longitude
        """
        return self.location

    def get_address(self):
        """
        Get address

        usage:
            string = get_address()

        :return address: address (string)
        """
        return self.address

    def get_area(self):
        """
        Get area dictionary with min, most_likely, max, mean, sd, pdf values

        usage:
            dictionary = get_area()

        :return area: a dictionary with min, most_likely, max, mean, sd, pdf values
        """
        return self.area

    def get_thickness(self):
        """
        Get thickness dictionary with min, most_likely, max, mean, sd, pdf values

        usage:
            dictionary = get_thickness()

        :return thickness: a dictionary with min, most_likely, max, mean, sd, pdf values
        """
        return self.thickness

    def get_volume(self):
        """
        Get volume dictionary with min, most_likely, max, mean, sd, pdf values

        usage:
            dictionary = get_volume()

        :return volume: a dictionary with min, most_likely, max, mean, sd, pdf values
        """
        return self.volume

    def set_name(self, name='string'):
        """
        Set name as string

        usage:
            set_name('name_of_instance')
        """
        self.name = name

    def set_location(self, **kwargs):
        """
        Set location values (lat, lon)

        usage:
            set_location(lat=10.0, lon=20.0)
        """
        self.set_values_to_variables(self.location, kwargs)

    def set_address(self, address='string'):
        """
        Set address as string

        usage:
            set_address(address='string')
        """
        self.address = address

    def set_area(self, **kwargs):
        """
        Set area values [km2], use the follow syntax:

        usage:
            set_area(min=10.0, most_likely=20.0, max=30.0, mean=40.0, sd=50.0, pdf='C')
        """
        self.set_values_to_variables(self.area, kwargs)

    def set_thickness(self, **kwargs):
        """
        Set thickness values [m], use the follow syntax:

        usage:
            set_thickness(min=10.0, most_likely=20.0, max=30.0, mean=40.0, sd=50.0, pdf='C')
        """
        self.set_values_to_variables(self.thickness, kwargs)

    def set_volume(self, **kwargs):
        """
        Set volume values [km3], use the follow syntax:

        usage:
            set_volume(min=10.0, most_likely=20.0, max=30.0, mean=40.0, sd=50.0, pdf='C')
        """
        self.set_values_to_variables(self.volume, kwargs)

    def __str__(self):
        """
        Print name, location, address, area, thickness, and volume
        """
        if self.volume['pdf'] is not None:
            volume = self.volume['most_likely']
            return '{0}, Lat: {1} Lon: {2}, {3}, and {4} km3 '.format(self.name,
                                                                      str(self.location['lat']),
                                                                      str(self.location['lon']),
                                                                      self.address, str(volume))

        elif self.area['pdf'] is not None and self.thickness['pdf'] is not None:
            volume = self.area['most_likely'] * self.thickness['most_likely']
            return '{0}, Lat: {1} Lon: {2}, {3}, and {4} km3 '.format(self.name,
                                                                      str(self.location[
                                                                              'lat']),
                                                                      str(self.location[
                                                                              'lon']),
                                                                      self.address,
                                                                      str(volume))
        else:
            return "Warning: The volume property  has not defined."


class Thermodynamic(Reservoir):
    """
    **Thermodynamics properties abstraction**

    :param name: string
    :param location: dictionary lat and lon
    :param address: string
    :param area: km^2
    :param thickness: m
    :param volume: km3
    :param reservoir_temp: ºC
    :param abandon_temp: ºC
    :param porosity: %
    :param rock_specific_heat: kJ/kg-ºC
    :param fluid_specific_heat: kJ/kg-ºC
    :param rock_density: kg/m3
    :param fluid_density: kg/m3
    """
    def __init__(self, **kwargs):
        self.reservoir_temp = {'min': 1.0, 'most_likely': 1.0, 'max': 1.0, 'mean': 1.0,
                               'sd': 1.0, 'pdf': 'C'}
        self.abandon_temp = {'min': 1.0, 'most_likely': 1.0, 'max': 1.0, 'mean': 1.0,
                             'sd': 1.0, 'pdf': 'C'}
        self.porosity = {'min': 1.0, 'most_likely': 1.0, 'max': 1.0, 'mean': 1.0,
                         'sd': 1.0, 'pdf': 'C'}
        self.rock_specific_heat = {'min': 1.0, 'most_likely': 1.0, 'max': 1.0, 'mean': 1.0,
                                   'sd': 1.0, 'pdf': 'C'}
        self.fluid_specific_heat = {'min': 1.0, 'most_likely': 1.0, 'max': 1.0,
                                    'mean': 1.0, 'sd': 1.0, 'pdf': 'C'}
        self.rock_density = {'min': 1.0, 'most_likely': 1.0, 'max': 1.0, 'mean': 1.0,
                             'sd': 1.0, 'pdf': 'C'}
        self.fluid_density = {'min': 1.0, 'most_likely': 1.0, 'max': 1.0, 'mean': 1.0,
                              'sd': 1.0, 'pdf': 'C'}
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
        super(Thermodynamic, self).__init__(**kwargs)

    def get_reservoir_temp(self):
        """
        To get reservoir_temp dictionary in [ºC] with min, most_likely, max, mean, sd, pdf values

        usage:
            dictionary = get_reservoir_temp()

        :return reservoir_temp: a dictionary with min, most_likely, max, mean, sd, pdf values
        """
        return self.reservoir_temp

    def get_abandon_temp(self):
        """
        To get abandon_temp dictionary in [ºC] with min, most_likely, max, mean, sd, pdf values

        usage:
            dictionary = get_abandon_temp()

        :return abandon_temp: a dictionary with min, most_likely, max, mean, sd, pdf values
        """
        return self.abandon_temp

    def get_porosity(self):
        """
        To get porosity dictionary in [%] with min, most_likely, max, mean, sd, pdf values

        usage:
            dictionary = get_porosity()

        :return porosity: a dictionary with min, most_likely, max, mean, sd, pdf values
        """
        return self.porosity

    def get_rock_specific_heat(self):
        """
        To get rock_specific_heat dictionary in [kJ/kg-ºC] with min, most_likely, max, mean, sd, pdf values

        usage:
            dictionary = get_rock_specific_heat()

        :return rock_specific_heat: a dictionary with min, most_likely, max, mean, sd, pdf values
        """
        return self.rock_specific_heat

    def get_rock_density(self):
        """
        To get rock_density dictionary in [kg/m^3] with min, most_likely, max, mean, sd, pdf values

        usage:
            dictionary = get_rock_density()

        :return rock_density: a dictionary with min, most_likely, max, mean, sd, pdf values
        """
        return self.rock_density

    def get_fluid_specific_heat(self):
        """
        To get fluid_specific_heat dictionary in [kJ/kg-ºC] with min, most_likely, max, mean, sd, pdf values

        usage:
            dictionary = get_fluid_specific_heat()

        :return fluid_specific_heat: a dictionary with min, most_likely, max, mean, sd, pdf values
        """
        return self.fluid_specific_heat

    def get_fluid_density(self):
        """
        To get fluid_density dictionary in [kg/m^3] with min, most_likely, max, mean, sd, pdf values

        usage:
            dictionary = get_fluid_density()

        :return fluid_density: a dictionary with min, most_likely, max, mean, sd, pdf values
        """
        return self.fluid_density

    def set_reservoir_temp(self, **kwargs):
        """
        Set reservoir_temp values [ºC], use the follow syntax:

        usage:
            set_reservoir_temp(min=10.0, most_likely=20.0, max=30.0, mean=40.0, sd=50.0, pdf='C')
        """
        self.set_values_to_variables(self.reservoir_temp, kwargs)

    def set_abandon_temp(self, **kwargs):
        """
        Set abandon_temp values [ºC], use the follow syntax:

        usage:
            set_abandon_temp(min=10.0, most_likely=20.0, max=30.0, mean=40.0, sd=50.0, pdf='C')
        """
        self.set_values_to_variables(self.abandon_temp, kwargs)

    def set_porosity(self, **kwargs):
        """
        Set porosity values [%], use the follow syntax:

        usage:
            set_porosity(min=10.0, most_likely=20.0, max=30.0, mean=40.0, sd=50.0, pdf='C')
        """
        self.set_values_to_variables(self.porosity, kwargs)

    def set_rock_specific_heat(self, **kwargs):
        """
        Set rock_specific_heat values [kJ/kg-ºC], use the follow syntax:

        usage:
            set_rock_specific_heat(min=10.0, most_likely=20.0, max=30.0, mean=40.0, sd=50.0,
                               pdf='C')
        """
        self.set_values_to_variables(self.rock_specific_heat, kwargs)

    def set_rock_density(self, **kwargs):
        """
        Set rock_density values [kg/m^3], use the follow syntax:

        usage:
            set_rock_density(min=10.0, most_likely=20.0, max=30.0, mean=40.0, sd=50.0, pdf='C')
        """
        self.set_values_to_variables(self.rock_density, kwargs)

    def set_fluid_specific_heat(self, **kwargs):
        """
        Set fluid_specific_heat values [kJ/kg-ºC], use the follow syntax:

        usage:
            set_fluid_specific_heat(min=10.0, most_likely=20.0, max=30.0, mean=40.0, sd=50.0,
                                pdf='C')
        """
        self.set_values_to_variables(self.fluid_specific_heat, kwargs)

    def set_fluid_density(self, **kwargs):
        """
        Set fluid_density values [kg/m^3], use the follow syntax:

        usage:
            set_fluid_density(min=10.0, most_likely=20.0, max=30.0, mean=40.0, sd=50.0, pdf='C')
        """
        self.set_values_to_variables(self.fluid_density, kwargs)

    @staticmethod
    def liquid_dominant_volumetric_energy(tr, ta, phi, cr, cf, rho_r, rho_f, a=None, h=None, v=None):
        """
        To calculate energy available in the reservoir by volumetric method in liquid dominated [kJ]
        """
        q = 1.0
        if v is not None:
            q = (rho_r * cr * (1.0 - phi) + rho_f * cf * phi) * (v * 1.0e9) * (tr - ta)
        elif a is not None and h is not None:
            q = (rho_r * cr * (1.0 - phi) + rho_f * cf * phi) * (a * 1.0e6) * h * (tr - ta)
        return q

    @staticmethod
    def two_phase_dominant_volumetric_energy(tr, ta, phi, cr, cf, rho_r, rho_f, a, h, v):
        """
        calculate energy available [kJ]

        HINT: This method has not been implemented yet. It uses the same equation as LQDE
        method (Liquid dominated volumetric energy) has.
        """
        q = 1.0
        if v is not None:
            q = (rho_r * cr * (1.0 - phi) + rho_f * cf * phi) * (v * 1.0e9) * (tr - ta)
        elif a is not None and h is not None:
            q = (rho_r * cr * (1.0 - phi) + rho_f * cf * phi) * (a * 1.0e6) * h * (tr - ta)
        return q


class GeothermalPowerPlant(Thermodynamic):
    """
    **Class for calculating power assessment of geothermal field**

    General variables:
        :param name: string
        :param location: dictionary lat and lon
        :param address: string
        :param area: km^2
        :param thickness: m
        :param volume: km3

    Thermodynamics properties:
        :param reservoir_temp: ºC
        :param abandon_temp: ºC
        :param porosity: %
        :param rock_specific_heat: kJ/kg-ºC
        :param fluid_specific_heat: kJ/kg-ºC
        :param rock_density: kg/m^3
        :param fluid_density: kg/m^3

    Power Plant characteristics:
        :param recovery_factor: heat recovery factor from heat source to power plant
        :param conversion_efficiency: efficiency of electrical conversion
        :param plant_net_capacity_factor: plant net capacity factor
        :param lifespan: lifespan years

    HINT: variable power_potential has Monte Carlo simulation results:
    percentiles list has values from 5% to 95%

    self.power_potential = {'base': 1.0, 'pdf': 1.0, 'iterations': 10000, 'statistics': \
    {'p_base': 1.0, 'mean': 1.0, 'sd': 1.0, 'skew': 1.0, 'kurt': 1.0, 'min': 1.0, 'max': 1.0, \
    'percentiles': []}}
    """
    def __init__(self, **kwargs):
        self.conversion_efficiency = {'min': 1.0, 'most_likely': 1.0, 'max': 1.0,
                                      'mean': 1.0, 'sd': 1.0, 'pdf': 'C'}
        self.recovery_factor = {'min': 1.0, 'most_likely': 1.0, 'max': 1.0,
                                'mean': 1.0, 'sd': 1.0, 'pdf': 'C'}
        self.plant_net_capacity_factor = {'min': 1.0, 'most_likely': 1.0, 'max': 1.0,
                                          'mean': 1.0, 'sd': 1.0, 'pdf': 'C'}
        self.lifespan = {'min': 1.0, 'most_likely': 1.0, 'max': 1.0,
                         'mean': 1.0, 'sd': 1.0, 'pdf': 'C'}
        self.power_potential = {'base': 1.0,
                                'pdf': 1.0,
                                'iterations': 10000,
                                'statistics': {'p_base': 1.0,
                                               'mean': 1.0,
                                               'sd': 1.0,
                                               'skew': 1.0,
                                               'kurt': 1.0,
                                               'min': 1.0,
                                               'max': 1.0,
                                               'percentiles': []}}
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
        super(GeothermalPowerPlant, self).__init__(**kwargs)

    def get_conversion_efficiency(self):
        """
        To get conversion_efficiency dictionary in [%] with min, most_likely, max, mean, sd, pdf values

        usage:
            dictionary = get_conversion_efficiency()

        :return conversion_efficiency: a dictionary with min, most_likely, max, mean, sd, pdf values
        """
        return self.conversion_efficiency

    def get_recovery_factor(self):
        """
        To get recovery_factor dictionary in [%] with min, most_likely, max, mean, sd, pdf values

        usage:
            dictionary = get_recovery_factor()

        :return recovery_factor: a dictionary with min, most_likely, max, mean, sd, pdf values
        """
        return self.recovery_factor

    def get_plant_net_capacity_factor(self):
        """
        To get plant_net_capacity_factor dictionary in [%] with min, most_likely, max, mean, sd, pdf values

        usage:
            dictionary = get_plant_net_capacity_factor()

        :return plant_net_capacity_factor: a dictionary with min, most_likely, max, mean, sd, pdf values
        """
        return self.plant_net_capacity_factor

    def get_lifespan(self):
        """
        To get lifespan dictionary in [%] with min, most_likely, max, mean, sd, pdf values

        usage:
            dictionary = get_lifespan()

        :return lifespan: a dictionary with min, most_likely, max, mean, sd, pdf values
        """
        return self.lifespan

    def set_conversion_efficiency(self, **kwargs):
        """
        Set conversion_efficiency values [%], use the follow syntax:

        usage:
            set_conversion_efficiency(min=10.0, most_likely=20.0, max=30.0, mean=40.0, sd=50.0, pdf='C')
        """
        self.set_values_to_variables(self.conversion_efficiency, kwargs)

    def set_recovery_factor(self, **kwargs):
        """
        Set recovery_factor values [%], use the follow syntax:

        usage:
            set_recovery_factor(min=10.0, most_likely=20.0, max=30.0, mean=40.0, sd=50.0, pdf='C')
        """
        self.set_values_to_variables(self.recovery_factor, kwargs)

    def set_plant_net_capacity_factor(self, **kwargs):
        """
        Set plant_net_capacity_factor values [%], use the follow syntax:

        usage:
            set_plant_net_capacity_factor(min=10.0, most_likely=20.0, max=30.0, mean=40.0, sd=50.0, pdf='C')
        """
        self.set_values_to_variables(self.plant_net_capacity_factor, kwargs)

    def set_lifespan(self, **kwargs):
        """
        Set lifespan values [years], use the follow syntax:

        usage:
            set_lifespan(min=10.0, most_likely=20.0, max=30.0, mean=40.0, sd=50.0, pdf='C')
        """
        self.set_values_to_variables(self.lifespan, kwargs)

    def power_energy(self, tr, ta, phi, cr, cf, rho_r, rho_f, rf, ce, pf, t, a, h, v, rtype='ld'):
        """
        **return power energy assessment [We]**

        :param t: lifespan [years]
        :param pf: plant net capacity factor [%]
        :param ce: conversion efficiency [%]
        :param rf: recovery factor [%]
        :param rho_f: rock density [kg/m^3]
        :param rho_r: fluid density [kg/m^3]
        :param cf: fluid specific heat [kJ/kg-ºC]
        :param cr: rock specific heat [kJ/kg-ºC]
        :param phi: porosity [%]
        :param h: thickness [m]
        :param ta: abandon temperature [ºC]
        :param tr: reservoir temperature [ºC]
        :param a: area [km^2]
        :param v: volume [km^3]
        :param rtype: ld = liquid dominant, tpd = two phase dominant
        :return: power energy [We]
        """
        q = 1.0
        if v is not None:
            if rtype == 'ld':
                a = None
                h = None
                q = self.liquid_dominant_volumetric_energy(tr, ta, phi, cr, cf, rho_r,
                                                           rho_f, a, h, v)
            elif rtype == 'tpd':
                a = None
                h = None
                q = self.two_phase_dominant_volumetric_energy(tr, ta, phi, cr, cf, rho_r,
                                                              rho_f, a, h, v)
        else:
            if rtype == 'ld':
                v = None
                q = self.liquid_dominant_volumetric_energy(tr, ta, phi, cr, cf, rho_r, rho_f, a, h, v)
            elif rtype == 'tpd':
                v = None
                q = self.two_phase_dominant_volumetric_energy(tr, ta, phi, cr, cf, rho_r,
                                                              rho_f, a, h, v)
        return (q * 1000) * rf * ce / (pf * (t * 31557600))

    def __str__(self):
        """
        To generate a table with all geothermal power plant values
        """
        if type(self.power_potential['pdf']) is mc.UncertainFunction:
            text = "Most Likely PowerGeneration: {0} [We]\n" \
                   "P10%: {1} [We]".format(self.power_potential['base'],
                                           self.power_potential['statistics']['percentiles'][1]
                                           )
        else:
            text = 'Reservoir simulation has not been done!'

        def get_values_ordered(val):
            keys = ['min', 'most_likely', 'max', 'mean', 'sd', 'pdf']
            return [val[keys[i]] for i in [0, 1, 2, 3, 4, 5]]

        table1 = BeautifulTable(max_width=80)
        table1.column_headers = ["name", "lat [oC]", "lon [oC]"]
        table1.append_row([self.name, self.location['lat'], self.location['lon']])
        table2 = BeautifulTable(max_width=110)
        table2.top_border_char = '='
        table2.header_separator_char = '='
        table2.bottom_border_char = '='
        table2.column_headers = ['Item', 'Variable', 'Symbol', 'Units', 'Min', 'Most_Likely',
                                 'Max', 'Mean', 'SD', 'PDF']
        table2.append_row([0, 'area', 'A', 'km2'] + get_values_ordered(self.area))
        table2.append_row([1, 'thickness', 'h', 'm'] + get_values_ordered(self.thickness))

        table2.append_row([2, 'volume', 'v', 'km3'] + get_values_ordered(self.volume))

        table2.append_row([3, 'reservoir_temp', 'Tr', 'oC'] +
                          get_values_ordered(self.reservoir_temp))
        table2.append_row([4, 'abandon_temp', 'Ta', 'oC'] +
                          get_values_ordered(self.abandon_temp))
        table2.append_row([5, 'porosity', 'phi', '%'] + get_values_ordered(self.porosity))
        table2.append_row([6, 'rock_specific_heat', 'Cr', 'kJ/kg-oC'] +
                          get_values_ordered(self.rock_specific_heat))
        table2.append_row([7, 'fluid_specific_heat', 'Cf', 'kJ/kg-oC'] +
                          get_values_ordered(self.fluid_specific_heat))
        table2.append_row([8, 'rock_density', 'rho_r', 'kg/m3'] +
                          get_values_ordered(self.rock_density))
        table2.append_row([9, 'fluid_density', 'rho_f', 'kg/m3'] +
                          get_values_ordered(self.fluid_density))
        table2.append_row([10, 'recovery_factor', 'RF', '%'] +
                          get_values_ordered(self.recovery_factor))
        table2.append_row([11, 'conversion_efficiency', 'Ce', '%'] +
                          get_values_ordered(self.conversion_efficiency))
        table2.append_row([12, 'plant_net_capacity_factor', 'Pf', '%'] +
                          get_values_ordered(self.plant_net_capacity_factor))
        table2.append_row([13, 'lifespan', 't', 'years'] + get_values_ordered(self.lifespan))
        return table1.get_string() + '\n' + table2.get_string() + '\n' + text


class MonteCarloSimulation(object):
    """
    Run Monte Carlo Simulation for figuring out Geothermal Power Energy
    """
    def __init__(self, **kwargs):
        self.iterations = 10000
        self.calc_time = 0.0

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

    def get_iterations(self):
        """
        To get iterations value

        usage:
            val = get_iterations()
        """
        return self.iterations

    def set_iterations(self, iterations=10000):
        """
        To set iterations value

        usage:
            set_iterations(10000)
        """
        self.iterations = iterations

    def probability_distribution_function(self, val, lognormal_adjust=False):
        """
        Probability Distribution Function

        :param lognormal_adjust: True return log(lognormal_values),
                False return lognormal_values
        :param val: variable that is a dictionary with {'min': 1.0, 'most_likely': 1.0,
                'max': 1.0, 'mean': 1.0, 'sd': 1.0, 'pdf': 'C'}
        :return pdf: probability distribution values a collection of trials for every PDF,
               Monte Carlo error propagation type
        """

        def my_log_norm(mu, sigma, tag=None):
            """
            A Log-Normal random variate

            Parameters
            ----------
            mu : scalar
                The location parameter
            sigma : scalar
                The scale parameter (must be positive and non-zero)
            tag : none
            """
            assert sigma > 0, 'Log-Normal "sigma" must be positive'
            return mc.uv(ss.lognorm(s=sigma, scale=mc.umath.exp(mu)), tag=tag)

        mc.npts = self.iterations
        pdf = val['pdf']

        if pdf == 'C':
            return val['most_likely']
        elif pdf == 'T':
            return mc.Triangular(val['min'], val['most_likely'], val['max'])
        elif pdf == 'U':
            return mc.Uniform(val['min'], val['max'])
        elif pdf == 'N':
            return mc.Normal(val['mean'], val['sd'])
        elif pdf == 'L':
            if lognormal_adjust:
                return mc.umath.log(my_log_norm(val['mean'], val['sd']))
            else:
                return my_log_norm(val['mean'], val['sd'])

    def calc_energy_potential(self, gpp, rtype='ld'):
        """
        Calculus of Power Energy Assessment

        usage:
            gpp = sim.calc_energy_potential(gpp)

        :param rtype: reservoir type
        :param gpp: Geothermal power Plant object
        :return: a Geothermal power plant object
        """
        start = clock()
        if type(gpp) is GeothermalPowerPlant:
            tr = gpp.reservoir_temp['most_likely']
            ta = gpp.abandon_temp['most_likely']
            phi = gpp.porosity['most_likely']
            cr = gpp.rock_specific_heat['most_likely']
            cf = gpp.fluid_specific_heat['most_likely']
            rho_r = gpp.rock_density['most_likely']
            rho_f = gpp.fluid_density['most_likely']
            rf = gpp.recovery_factor['most_likely']
            ce = gpp.conversion_efficiency['most_likely']
            pf = gpp.plant_net_capacity_factor['most_likely']
            t = gpp.lifespan['most_likely']

            if gpp.volume['pdf'] is None:
                a = gpp.area['most_likely']
                h = gpp.thickness['most_likely']
                v = None
                gpp.power_potential['base'] = gpp.power_energy(tr, ta, phi, cr, cf, rho_r,
                                                               rho_f, rf, ce, pf, t, a, h, v, rtype)
            else:
                a = None
                h = None
                v = gpp.volume['most_likely']
                gpp.power_potential['base'] = gpp.power_energy(tr, ta, phi, cr, cf, rho_r,
                                                               rho_f, rf, ce, pf, t, a,
                                                               h, v, rtype)

            tr = self.probability_distribution_function(gpp.reservoir_temp)
            ta = self.probability_distribution_function(gpp.abandon_temp)
            if gpp.porosity['pdf'] == 'L':
                phi = self.probability_distribution_function(gpp.porosity,
                                                             lognormal_adjust=True)
            else:
                phi = self.probability_distribution_function(gpp.porosity)

            cr = self.probability_distribution_function(gpp.rock_specific_heat)
            cf = self.probability_distribution_function(gpp.fluid_specific_heat)
            rho_r = self.probability_distribution_function(gpp.rock_density)
            rho_f = self.probability_distribution_function(gpp.fluid_density)
            rf = self.probability_distribution_function(gpp.recovery_factor)
            ce = self.probability_distribution_function(gpp.conversion_efficiency)
            pf = self.probability_distribution_function(gpp.plant_net_capacity_factor)
            t = self.probability_distribution_function(gpp.lifespan)

            if gpp.volume['pdf'] is None:
                a = self.probability_distribution_function(gpp.area)
                h = self.probability_distribution_function(gpp.thickness)
                v = None
                gpp.power_potential['pdf'] = gpp.power_energy(tr, ta, phi, cr, cf, rho_r,
                                                              rho_f, rf, ce, pf, t, a, h, v, rtype)
            else:
                a = None
                h = None
                v = self.probability_distribution_function(gpp.volume)
                gpp.power_potential['pdf'] = gpp.power_energy(tr, ta, phi, cr, cf, rho_r,
                                                              rho_f, rf, ce, pf, t, a,
                                                              h, v, rtype)

            gpp.power_potential['iterations'] = self.iterations
            p_base = gpp.power_potential['pdf'] <= gpp.power_potential['base']
            gpp.power_potential['statistics']['p_base'] = 1.0 - p_base
            gpp.power_potential['statistics']['mean'] = gpp.power_potential['pdf'].mean
            gpp.power_potential['statistics']['sd'] = gpp.power_potential['pdf'].std
            gpp.power_potential['statistics']['skew'] = gpp.power_potential['pdf'].skew
            gpp.power_potential['statistics']['kurt'] = gpp.power_potential['pdf'].kurt
            gpp.power_potential['statistics']['min'] = gpp.power_potential['pdf'].percentile(0)
            gpp.power_potential['statistics']['max'] = gpp.power_potential['pdf'].percentile(1)
            gpp.power_potential['statistics']['percentiles'] = []
            for i in range(0, 19):
                val = 5 * i + 5
                percentile = gpp.power_potential['pdf'].percentile(val/100.0)
                gpp.power_potential['statistics']['percentiles'].append(percentile)
            self.calc_time = clock() - start

            if gpp.volume['pdf'] is not None:
                print('HINT: VOLUME WAS USED.\nSIMULATION ... DONE')
            elif gpp.area['pdf'] is not None and gpp.thickness['pdf'] is not None and gpp.volume['pdf'] is None:
                print('HINT: AREA AND THICKNESS WERE USED.\nSIMULATION ... DONE')
            else:
                print('ERROR: Distribution function must be valid: T, N, C, U, L\nSIMULATION ... FAILURE !\n')

            return gpp

    def __str__(self):
        """Print Iterations and calculation time"""
        return 'Iterations: {0}\nCalculation time: {1}s'.format(self.iterations,
                                                                self.calc_time)


class Tools(object):
    """
    Object with helping tools
    """
    def __init__(self, **kwargs):
        self.fn_input = 'input.csv'
        self.fn_output = 'output.csv'
        self.data_base = None
        self.num_figures = 6
        self.num_figures_plot = 3
        self.eng_format = 'M'
        self.hist_bins = 25
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

    def get_num_figures(self):
        """
        To get num_figures

        usage:
            val = get_num_figures()
        """
        return self.num_figures

    def get_num_figures_plot(self):
        """
        To get num_figures_plot

        usage:
            val = get_num_figures_plot()
        """
        return self.num_figures_plot

    def get_eng_format(self):
        """
        To get eng_format

        usage:
            val = get_eng_format()
        """
        return self.eng_format

    def get_hist_bins(self):
        """
        To get hist_bins

        usage:
            val = get_hist_bins()
        """
        return self.hist_bins

    def set_num_figures(self, num_figures=6):
        """
        To set num_figures

        usage:
            set_num_figures(6)
        """
        self.num_figures = num_figures

    def set_num_figures_plot(self, num_figures_plot=3):
        """
        To set num_figures_plot

        usage:
            set_num_figures_plot(2)
        """
        self.num_figures_plot = num_figures_plot

    def set_eng_format(self, eng_format='M'):
        """
        To set eng_format

        usage:
            set_eng_format('M')
        """
        self.eng_format = eng_format

    def set_hist_bins(self, hist_bins=25):
        """
        To set hist_bins

        usage:
            set_hist_bins(25)
        """
        self.hist_bins = hist_bins

    @staticmethod
    def figures_to_present(value=1.0, figures=1):
        """
        present the number by number of figures selected.
        return string
        """
        fi = '%.' + str(figures) + 'g'
        return fi % value

    def eng_fmt(self, value=1.0, option="1", *arg):
        """
        Present the data as Engineering format

        case1: eng_fmt(value=3538, option="k") -> return 3.538 float
        case2: eng_fmt(value=3538, option="k", 2) -> return 3.5 string
        """
        switcher = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "c": 1e-2, "1": 1.0e0,
                    "k": 1.0e3, "M": 1.0e6, "G": 1.0e9, "T": 1.0e12}
        if len(arg):
            figure = arg
            value = value / switcher.get(option, 1.0)
            return self.figures_to_present(value, figure[0])
        else:
            return value / switcher.get(option, 1.0)

    def print_results(self, gpp):
        """
        Print results

        :param gpp: Geothermal power plant object
        :return: nothing
        """
        if type(gpp.power_potential['pdf']) is not mc.UncertainFunction:
            print('Reservoir simulation has not been done!')
            return None

        base = gpp.power_potential['base']
        e = gpp.power_potential['pdf']
        ite = str(gpp.power_potential['iterations'])
        x = e <= base
        x = 1.0 - x
        x = self.eng_fmt(x, '1', self.num_figures)

        print("\nMAIN INFORMATION:")
        print("Most Likely PowerGeneration ["+self.eng_format+'We]='+self.eng_fmt(base, self.eng_format, self.num_figures))
        print("Probability of "+self.eng_fmt(base, self.eng_format, self.num_figures)+"[" + self.eng_format + 'We]='+ x)
        print("P10% ["+self.eng_format+'We]='+self.eng_fmt(e.percentile(0.1), self.eng_format, self.num_figures))
        print("\nSTATISTICAL ANALYSIS:")
        print("Iterations="+ite)
        print("Mean="+self.eng_fmt(e.mean, self.eng_format, self.num_figures)+"["+self.eng_format+'We]')
        print("Median="+self.eng_fmt(e.percentile(0.5), self.eng_format, self.num_figures)+"["+self.eng_format+'We]')
        print("Standard Deviation="+self.eng_fmt(e.std, self.eng_format, self.num_figures)+"["+self.eng_format+'We]')
        print("Skew="+self.figures_to_present(e.skew, self.num_figures))
        print("Kurt="+self.figures_to_present(e.kurt, self.num_figures))
        print("Minimum="+self.eng_fmt(e.percentile(0), self.eng_format, self.num_figures)+"["+self.eng_format+'We]')
        print("Maximum="+self.eng_fmt(e.percentile(1), self.eng_format, self.num_figures)+"["+self.eng_format+'We]')
        for i in range(0, 19):
            val = 5 * i + 5
            print("P"+str(val)+'%='+self.eng_fmt(e.percentile(val / 100.0), self.eng_format, self.num_figures)+
                "["+self.eng_format+'We]')
        print('\nEND')

    def read_file_csv(self, fn_input=None):
        """
        Read File coma separated value *.csv

        usage: gpp = tools.read_file_csv('file_name')

        :param fn_input: File name string
        :return gpp: Geothermal Power Plant object
        """
        if fn_input is not None:
            self.fn_input = fn_input
        fil = {'area': 0, 'thickness': 1, 'volume': 2, 'reservoir_temp': 3,
               'abandon_temp': 4, 'porosity': 5, 'rock_specific_heat': 6,
               'fluid_specific_heat': 7, 'rock_density': 8, 'fluid_density': 9,
               'recovery_factor': 10, 'conversion_efficiency': 11,
               'plant_net_capacity_factor': 12, 'lifespan': 13, 'name': 0, 'location': 0}
        col = np.dtype([
            ('item', str, 50),
            ('name', str, 50),
            ('lat', float),
            ('lon', float),
            ('variable', str, 50),
            ('symbol', str, 50),
            ('unit', str, 50),
            ('min', float),
            ('most_likely', float),
            ('max', float),
            ('mean', float),
            ('sd', float),
            ('pdf', str, 50)])
        self.data_base = np.genfromtxt(self.fn_input, delimiter=',', comments='#', dtype=col)
        gpp = GeothermalPowerPlant()
        gpp.name = self.data_base['name'][fil['name']]
        for key in gpp.location:
            gpp.location[key] = self.data_base[key][fil['location']]

        # ------ New ---------------
        for key in gpp.area:
            val = self.data_base[key][fil['area']]
            if val == 'None' or val == 'none':
                gpp.area[key] = None
            else:
                gpp.area[key] = val

        for key in gpp.thickness:
            val = self.data_base[key][fil['thickness']]
            if val == 'None' or val == 'none':
                gpp.thickness[key] = None
            else:
                gpp.thickness[key] = val

        for key in gpp.volume:
            val = self.data_base[key][fil['volume']]
            if val == 'None' or val == 'none':
                gpp.volume[key] = None
            else:
                gpp.volume[key] = val

        # ---- New finished ---------------

        for key in gpp.reservoir_temp:
            gpp.reservoir_temp[key] = self.data_base[key][fil['reservoir_temp']]
        for key in gpp.abandon_temp:
            gpp.abandon_temp[key] = self.data_base[key][fil['abandon_temp']]
        for key in gpp.porosity:
            gpp.porosity[key] = self.data_base[key][fil['porosity']]
        for key in gpp.rock_specific_heat:
            gpp.rock_specific_heat[key] = self.data_base[key][fil['rock_specific_heat']]
        for key in gpp.fluid_specific_heat:
            gpp.fluid_specific_heat[key] = self.data_base[key][fil['fluid_specific_heat']]
        for key in gpp.rock_density:
            gpp.rock_density[key] = self.data_base[key][fil['rock_density']]
        for key in gpp.fluid_density:
            gpp.fluid_density[key] = self.data_base[key][fil['fluid_density']]
        for key in gpp.conversion_efficiency:
            gpp.conversion_efficiency[key] = self.data_base[key][fil['conversion_efficiency']]
        for key in gpp.recovery_factor:
            gpp.recovery_factor[key] = self.data_base[key][fil['recovery_factor']]
        for key in gpp.plant_net_capacity_factor:
            gpp.plant_net_capacity_factor[key] = self.data_base[key][
                fil['plant_net_capacity_factor']]
        for key in gpp.lifespan:
            gpp.lifespan[key] = self.data_base[key][fil['lifespan']]
        print('READ FILE ... OK')
        return gpp

    def write_file_cvs(self, gpp, fn_output=None):
        """
        Write File coma separated value *.csv
        """
        if fn_output is not None:
            self.fn_output = fn_output
        output_file = open(self.fn_output, 'w')

        def get_values(val):
            keys = ['min', 'most_likely', 'max', 'mean', 'sd', 'pdf']
            string = ''
            for i in [0, 1, 2, 3, 4, 5]:
                st = str(val[keys[i]])
                if st == 'nan':
                    string += ','
                else:
                    string += ',' + st
            return string + '\n'

        text = '#Imput file for the Assessment of Power Energy Generation by using ' \
               'Volumetric Method and Monte Carlo Probabilistic analysis,' \
               ',,,,,,,,,,,\n#Item,Name_Place,Latitude,Longitude,Reservoir_Properties,' \
               'Symbol,Units,Min,Most_Likely,Max,Mean,Standard_Deviation,Distribution_Type\n'
        text += '0,'+gpp.name+','+str(gpp.location['lat'])+','+str(gpp.location['lon']) + \
                ',area,A,km2'+get_values(gpp.area)
        text += '1,,,,thickness,h,m'+get_values(gpp.thickness)

        text += '2,,,,volume,v,km3'+get_values(gpp.volume)

        text += '3,,,,reservoir_temp,Tr,oC' + get_values(gpp.reservoir_temp)
        text += '4,,,,abandon_temp,Ta,oC' + get_values(gpp.abandon_temp)
        text += '5,,,,porosity,phi,%' + get_values(gpp.porosity)
        text += '6,,,,rock_specific_heat,Cr,kJ/kg-oC' + get_values(gpp.rock_specific_heat)
        text += '7,,,,fluid_specific_heat,Cf,kJ/kg-oC' + get_values(gpp.fluid_specific_heat)
        text += '8,,,,rock_density,rho_r,kg/m3' + get_values(gpp.rock_density)
        text += '9,,,,fluid_density,rho_f,kg/m3' + get_values(gpp.fluid_density)
        text += '10,,,,recovery_factor,Rf,%' + get_values(gpp.recovery_factor)
        text += '11,,,,conversion_efficiency,Ce,%' + get_values(gpp.conversion_efficiency)
        text += '12,,,,plant_net_capacity_factor,Pf,%' + \
                get_values(gpp.plant_net_capacity_factor)
        text += '13,,,,lifespan,t,years' + get_values(gpp.lifespan)
        output_file.write(text)
        output_file.close()
        print('WRITE TO FILE ... OK')

    @staticmethod
    def set_backend_figure(backend='Qt4Agg'):
        """
        Set matplotlib backend for improving the figure presentation
        By default, python uses TkAgg and ipython uses inline. this function has Qt4Agg.
        Use this function only when you need change backend otherwise refrain to use it.
        at that time to change the backend, restart python kernel.
        """
        try:
            plt.switch_backend(backend)
        except ValueError:
            print('{0} Not found. Default has set up'.format(backend))
        except ImportError:
            print('{0} Not found. Default has set up'.format(backend))

    def plot_pdf(self, gpp, type_graph='hist', show=False, x_lim=None, hist_fitted_curve=True,
                 hist_edge_color='g', hist_line_width=0, hist_type='stepfilled', fig_dpi=None,
                 fig_size=None, axis_label_size=10, tick_params_size=8, text_size=8,
                 title_size=10, legend_text_size=10):
        """
        Plot probability distribution function

        :param axis_label_size: axis label size (int)
        :param tick_params_size: tick size (int)
        :param text_size: text size (int)
        :param title_size: title font size (int)
        :param legend_text_size: legend font size (int)
        :param fig_size: figure size, tuple fig_sie=(width, height)
        :param fig_dpi: figure dpi, integer
        :param hist_type: histogram type
        :param hist_line_width: line width in histogram
        :param hist_edge_color: show edge in histogram
        :param hist_fitted_curve: show fitted curve in histogram
        :param type_graph: 'hist' = hist, 'lower' = Cumulative Relative Frequency (lower),
                'higher' = Cumulative relative frequency (higher),
                'linear' = linear plot summary
        :param gpp: Geothermal power plant object
        :param show: if True will show figure
        :param x_lim: tuple x_lim=(min,max)
        :return: plot
        """
        if type_graph is 'hist':
            type_graph = False
        elif type_graph is 'lower':
            type_graph = True
        elif type_graph is 'higher':
            type_graph = -1
        elif type_graph is 'linear':
            pass
        else:
            type_graph = False
        if type(gpp.power_potential['pdf']) is not mc.UncertainFunction:
            print('Reservoir simulation has not been done!')
            return None
        num_div = self.eng_fmt(1, self.eng_format)
        if type_graph is 'linear':
            if fig_size is None:
                fig_size = (10, 1)
            plt.figure(figsize=fig_size, facecolor='w', dpi=fig_dpi)
            plt.tick_params(
                axis='y',
                which='both',
                left='off',
                right='off',
                labelleft='off',
                labelright='off'
            )
            plt.tick_params(
                axis='x',
                direction='out',
                top='off',
                labelsize=tick_params_size
            )
            p_base_loc = 100 - gpp.power_potential['statistics']['p_base'] * 100.0
            base = gpp.power_potential['base']
            y = gpp.power_potential['statistics']['percentiles']
            plt.barh(1, 5, left=0, color='#228B22', align='center', edgecolor='none')
            plt.barh(1, 5, left=5, color='#228B22', align='center', edgecolor='none')
            plt.barh(1, p_base_loc - 10, left=10, color='#FFA500', align='center',
                     edgecolor='none')
            plt.barh(1, 95 - p_base_loc, left=p_base_loc, color='#FF6347', align='center',
                     edgecolor='none')
            plt.barh(1, 5, left=95, color='#FF6347', align='center', edgecolor='none')
            x_ticks_label = [self.eng_fmt(y[0], self.eng_format, self.num_figures_plot),
                             self.eng_fmt(y[1], self.eng_format, self.num_figures_plot),
                             self.eng_fmt(base, self.eng_format, self.num_figures_plot),
                             self.eng_fmt(y[18], self.eng_format, self.num_figures_plot)]
            plt.xticks([5, 10, p_base_loc, 95], x_ticks_label)
            plt.xlabel("Power Generation [" + self.eng_format + "We]",
                       fontsize=axis_label_size)
            plt.xlim(0, 100)
            plt.text(4, 1, '95%')
            plt.text(9, 1, '90%')
            plt.text(p_base_loc - 2, 1, self.eng_fmt(100 - p_base_loc, '1', 3) + '%')
            plt.text(94, 1, '5%')
            plt.text(4, 0.8, 'Proven reserves')
            plt.text(p_base_loc - 8, 0.8, 'Most likely reserves')
            plt.text(82, 0.8, 'Maximum reserves')
            if show:
                self.show_pdf()
            return None

        base = gpp.power_potential['base']
        e = gpp.power_potential['pdf']
        ite = gpp.power_potential['iterations']

        fig = plt.figure(facecolor='w', dpi=fig_dpi, figsize=fig_size)
        ax = fig.add_subplot(111)

        mcpts_data = [i * num_div for i in e._mcpts]

        if x_lim is None:
            r_min = min(mcpts_data)
            r_max = max(mcpts_data)
        else:
            r_min = x_lim[0]
            r_max = x_lim[1]
        if not type_graph:
            # ax.hist(mcpts_data, bins=self.hist_bins, normed=True, label='Estimated',
            ax.hist(mcpts_data, bins=self.hist_bins, density=True, label='Estimated',
                    color='#3cb371', cumulative=type_graph, range=(r_min, r_max),
                    histtype=hist_type, edgecolor=hist_edge_color,
                    linewidth=hist_line_width)  # Probability
            if hist_fitted_curve:
                p = ss.kde.gaussian_kde(mcpts_data)
                xp = np.linspace(r_min, r_max, 100)
                ax.plot(xp, p.evaluate(xp), color='r', label='Fitted', linewidth=2)
        else:
            # ax.hist(mcpts_data, bins=self.hist_bins, normed=True, label='Estimated', color='r',
            ax.hist(mcpts_data, bins=self.hist_bins, density=True, label='Estimated', color='r',
                    cumulative=type_graph, range=(r_min, r_max), histtype='step')
        figure01_ymin, figure01_ymax = ax.get_ylim()
        ax.tick_params(axis='x', labelsize=tick_params_size)
        ax.tick_params(axis='y', labelsize=tick_params_size)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.4g'))
        p05 = e.percentile(0.05) * num_div
        ax.axvline(p05, color='#FFA500', linestyle='--', linewidth=2)
        ax.text(p05, figure01_ymax * 0.9,
                'P5=' + self.eng_fmt(p05, '1', self.num_figures_plot) + "[" +
                self.eng_format + 'We]', fontsize=text_size, horizontalalignment='center')
        p10 = e.percentile(0.1) * num_div
        ax.axvline(p10, color='r', linestyle='--', linewidth=2)
        ax.text(p10, figure01_ymax * 0.05,
                'P10=' + self.eng_fmt(p10, '1', self.num_figures_plot) + "[" +
                self.eng_format + 'We]', fontsize=text_size, horizontalalignment='left',
                color='k')
        p95 = e.percentile(0.95) * num_div
        ax.axvline(p95, color='y', linestyle='--', linewidth=2)
        ax.text(p95, figure01_ymax * 0.2,
                'P95=' + self.eng_fmt(p95, '1', self.num_figures_plot) + "[" +
                self.eng_format + 'We]', fontsize=text_size, horizontalalignment='center')
        p_base = base * num_div
        x = self.eng_fmt(100 - 100 * (e <= base), '1', self.num_figures_plot)
        ax.axvline(p_base, color='b', linestyle='--', linewidth=2)
        ax.text(p_base, figure01_ymax * 0.5,
                'P(' + self.eng_fmt(p_base, '1', self.num_figures_plot) + self.eng_format +
                'We)=' + x + '%', fontsize=text_size, horizontalalignment='left')
        ax.set_title("Power Energy Available for " + str(gpp.lifespan['most_likely']) +
                     ' years' + '\n' + gpp.name + '. Iterations: ' + str(ite),
                     fontsize=title_size)
        ax.set_xlabel("Power Generation [" + self.eng_format + "We]", fontsize=axis_label_size)
        if type_graph == -1:
            ax.set_ylabel("Cumulative Relative Frequency [higher than]",
                          fontsize=axis_label_size)
        elif type_graph:
            ax.set_ylabel("Cumulative Relative Frequency [lower than]",
                          fontsize=axis_label_size)
        elif not type_graph:
            ax.set_ylabel(r"F(x)=100$\Delta$xFreq", fontsize=axis_label_size)
        ax.grid(linestyle=':')
        ax.legend(loc=1, fontsize=legend_text_size)
        if show:
            self.show_pdf()

    @staticmethod
    def show_pdf():
        plt.show()

# --------------------------------------
# RUN SIMULATION
# To import beta-library from specific folder:
# In[1]: import sys
# In[2]: sys.path.append('C:\Dropbox\PycharmProjects\gppeval_beta')
# In[3]: import gppeval
# In[4]: gppeval.__version__
# Out[5]: '2018.10.11.0.1.dev1'

if __name__ == "__main__":
    # Test: Input data manually
    tools = Tools()
    sim = MonteCarloSimulation()
    plant = GeothermalPowerPlant()
    plant.set_name('Nombre de Jesus. El Salvador')
    plant.set_location(lat=14, lon=-88.73)
    plant.set_area(min=5, most_likely=6, max=7, mean=0, sd=0, pdf='T', doc=5)
    plant.set_thickness(min=450, most_likely=500, max=600, mean=0.0, sd=0.0, pdf='T')
    plant.set_volume(pdf=None)
    plant.set_reservoir_temp(min=130, most_likely=160, max=163, mean=0.0, sd=0.0, pdf='T')
    plant.set_abandon_temp(min=0.0, most_likely=80, max=0.0, mean=0.0, sd=0.0, pdf='C')
    plant.set_porosity(min=0.0, most_likely=0.06, max=0.0, mean=0.06, sd=0.02, pdf='L')
    plant.set_rock_specific_heat(min=0.85, most_likely=0.85, max=0.9, mean=0.0, sd=0.0,
                                 pdf='T')
    plant.set_fluid_specific_heat(min=0, most_likely=5.18, max=0, mean=0.0, sd=0.0, pdf='C')
    plant.set_rock_density(min=0, most_likely=2500, max=0, mean=0.0, sd=0.0, pdf='C')
    plant.set_fluid_density(min=0, most_likely=764.45, max=0, mean=0.0, sd=0.0, pdf='C')
    plant.set_recovery_factor(min=0.0, most_likely=0.23, max=0.0, mean=0.0, sd=0.0, pdf='C')
    plant.set_conversion_efficiency(min=0.1, most_likely=0.12, max=0.2, mean=0.0, sd=0.0,
                                    pdf='T')
    plant.set_plant_net_capacity_factor(min=0.9, most_likely=0.95, max=1.0, mean=0.0, sd=0.0,
                                        pdf='T')
    plant.set_lifespan(min=0, most_likely=25, max=0, mean=0.0, sd=0.0, pdf='C')

    plant = sim.calc_energy_potential(plant)
    tools.print_results(plant)
