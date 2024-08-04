#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 19:24:11 2020

@author: Carlos Pocasangre
@email: carlos.pocasangre@ues.edu.sv
"""
import gppeval

def scenario(pp, x_lim, ite=10000):
    tool = gppeval.Tools()
    sim = gppeval.MonteCarloSimulation()
    sim.set_iterations(iterations=ite)
    pp = sim.calc_energy_potential(pp, rtype='tpd')
    print(pp)
    tool.print_results(pp)
    tool.plot_pdf(pp, show=True, x_lim=x_lim, hist_line_width=0.5, hist_type='bar')


test = gppeval.Tools().read_file_csv(fn_input='reservoir_properties_list.csv')

# scenario 1 (25 years)
scenario(test, [20, 120])

# scenario 2 (50 years)
test.set_lifespan(most_likely=50.0)
scenario(test, [10, 60])
