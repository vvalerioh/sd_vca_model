# -*- coding: utf-8 -*-
"""
Created on Sat Jun  17 01:46:08 2023

@author: vvalerioh and Greg Kiker
"""

import csv
import numpy as np
import sympy as sp
from sympy import symbols, solve, exp, Eq
import math
import random



# Greg Kiker & Valerie Valerio version 3

runno = 0
# Make output text file and write in parameter names - names, order need to match the code at the end that writes in outputs
#with open('output1.txt', 'w', newline='') as f:
with open('output1_final.txt', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["run_no",
        "totalGoats",
        #"pctYoungGoats", #"pctAdultGoats",
        #"pctBreedingGoats",
        #"pctOldGoats",
        #"pctChangeGoatPop",
        #"accGoatsConsumed",
        #"accGoatsSold",
        #"accGoatDemand",
        "avgProducerProfit", #"avgProducerCost",
        #"accProducerProfit", "accProducerCost",
        "W",
        "P", "S", "D",
        #"avgGoatsFarmer",
        #"accYoungMortality", # percentage of all born goats that died while young
        #"accBreedingMortality", # percentage of all goats that died during breeding years
        "pctChangePrice",
        "pctDemandMet",
        "avgWeeklyGoatsSold"
    ])

    # Outputs to use for MC Filtering and validation

#with open('mcf_outputs.txt', 'w', newline='') as fi:
#with open('mcf_outputs_16traj.txt', 'w', newline='') as fi:
with open('mcf_outputs_final.txt', 'w', newline='') as fi:
    writer = csv.writer(fi)
    writer.writerow(["run_no",
                     "totalGoats",
                     "pctYoungGoats", "pctMaleGoats",
                     "pctBreedingGoats",
                     "pctOldGoats",
                     "pctChangeGoatPop",
                     "accGoatsConsumed",
                     "accGoatsSold",
                     "accGoatDemand",
                     "avgProducerProfit", "avgProducerCost",
                     "W",
                     "P", "S", "D",
                     "avgGoatsFarmer",
                     "accYoungMortality", # percentage of all born goats that died while young
                     "accBreedingMortality", # percentage of all goats that died during breeding years
                     "pctChangePrice",
                     "pctDemandMet",
                     "avgWeeklyGoatsSold"])

#with open('vc-results.csv', 'w', newline='') as file:
#with open('vc-results_16traj.csv', 'w', newline='') as file:
with open('vc-results_final.txt', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["runno", "X", "YoungGoats",
                     "BreedingGoats", "MaleGoats", "OldGoats", "TotalGoats",
                     "Demand", "Supply", "inventoryRatio", "W",
                     "InitialPrice", "Price",
                     "SellingGoats",
                     "ConsumingGoats",
                     "dYdt", # Young
                     "dBdt", # Breeding does
                     "dMdt", # Bucks
                     "dZdt", # Old
                     "dDdt", "DemandIntercept", "DemandSchedule",
                     "ProducerProfit",
                     "AccProducerProfit"])
#Open the input.dat file
#with open('eswatinivca_inputfactors_FacSample_8traj.sam', newline='') as infile:
#with open('eswatinivca_inputfactors_16traj_FacSample.sam', newline='') as infile:
with open('eswatinivca_inputfactors_final_FacSample.sam', newline='') as infile:
#with open('eswatinivca_inputfactors_FacSample_8traj.csv', newline='') as infile:
    inputreader = csv.reader(infile)
    #inputreader = csv.reader(infile)
    header = next(inputreader) # skips header line
    print(header)
    #   for row in inputreader:
    #       print(row[0])
    #       print(row[1])
    #Parameters Assigned from input.dat file
    for row in inputreader:
        runno = runno+1
        print(row)

        ageFirstKidding = float(row[0])  # weeks
        kiddingInterval = float(row[1])  # How often do kidding occur? from days to weeks
        kiddingRate = float(row[2])  # % of kids per doe each kidding
        ratioFemales = float(row[3])
        avgBreedingPeriod = float(row[4]) # float(row[4]) # 3.9 Years *52 = 202 weeks # Time during which goats are productive, 4 years
        timeLife = float(row[5])
        stillbirthRate = float(row[6]) # needs to be divided by gestation duration
        youngMortality = float(row[7]) #0#0.0053698586 #float(row[8]) # Fraction 1/yr = 0.28 1/yr = 0.0053698586 /wk
        breedingMortality = float(row[8]) # 0#0.0019178066  #float(row[10]) # Fraction 1/yr = 0.1 1/yr = 0.0019178066 /wk
        oldMortality = float(row[9])  # 0#0.0019178066  #float(row[11]) # Fraction 1/yr = 0.1 1/yr = 0.0019178066 /wk

        prodCost = float(row[10]) #  float(row[11]) #  115 # float(row[12]) # Production cost per goat (50 in Hamza paper)

        nFarmers = float(row[11])
        minHerdSize = float(row[12])

        demandAdjustmentTime = float(row[13])# float(row[14]) # 2.0 #float(row[16])  # weeks (check value 2) # Time it takes for demand to adjust
        perceivedInventoryDelay = float(row[14]) # float(row[15]) # 2.0 #float(row[17]) # weeks(check value 2) # Time it takes to perceive inventory

        priceElasticityInventoryRatio = float(row[15]) # float(row[17]) # -0.01 #float(row[19]) #(check value -0.3)
        priceElasticityDemand =  float(row[16]) # float(row[18]) # -0.01 #float(row[20]) #(check value -0.5)
        initialPrice = float(row[17]) # float(row[19]) # 830 #672 # From surveys in Rodriguez (2018)#830.0 #float(row[23])

        goatsPerFarmer = float(row[18]) # float(row[21]) # 15.8 #8.4#k = symbols('k', real = True) # 0.006136707191 # Calculated with W = kexp(k*P) for P = initialPrice

        # Herd structure
        pctKids = float(row[19])
        pctBreeding = float(row[20])
        pctBucks = float(row[21]) # Half of bucks are "young", half are old
        pctOld = 1 - pctKids - pctBreeding - pctBucks
        if pctOld < 0:
            pctOld = 0

        #pctSold = 0.41 # Mean between 2 available values
        #goatsSold = 3.25 # 3.25 mean of two possible values - unclear from Rodriguez
        #pctConsumed = 0.633 # mean of two values
        #goatsConsumed  = 3 # average of two values by Rodriguez
        initialDemand = float(row[22]) # 18 # 16.25 converted to weekly demand
        refConsuming = float(row[23])

        willingnessSell = float(row[24])  #float(row[37]) #check value 0.8     # Willingness to sell (between 1-0)
        demandShock = 0 # Allows for external shocks (or increased demand some months of the year)
        demandIntercept = np.log(initialDemand/np.power(initialPrice, priceElasticityDemand))*(1 + demandShock)
        demandSchedule = np.exp(demandIntercept)*np.power(initialPrice, priceElasticityDemand)
        inventoryGoal = float(row[25]) #W*nFarmers*goatsPerFarmer/(initialDemand + refConsuming)  # 211.895 # Minimum herd size / weekly demand calculated with minimum herd size

        # Initial Values
        W = willingnessSell
        totalGoats = nFarmers*goatsPerFarmer
        B = pctBreeding * totalGoats #1*totalGoats # 1053.0 #float(row[33]) #check value 150.0   # Breeding goats
        Y = pctKids * totalGoats #0.2*totalGoats #100.0  #float(row[34]) #check value 70.0     # Young goats
        M = pctBucks * totalGoats
        Z = pctOld * totalGoats #float(row[36])  #check value  60.0     # Old goats

        # Variables to save cumulative values
        ## Market
        accGoatsSold = 0
        accGoatsConsumed = 0
        accGoatDemand = 0

        ## Production
        accYoungPop = Y
        accBreedingPop = B
        accMalePop = M
        accYoungDeaths = 0
        accBreedingDeaths = 0
        accMaleDeaths = 0

        # Supply
        S = W * (M + Z + Y + B) - minHerdSize*nFarmers
        if S < 0:
            S = 1

        # Demand
        D = initialDemand

        P = initialPrice

        Mem = [S] * round(perceivedInventoryDelay)  #GAK

        #GAK Profit Variables...
        producerProfit = 0.0
        accProducerProfit = 0.0
        accProducerCost = 0.0

        #  Loop  in weeks 260 = 5 years, 390 = 7.5, 520 = 10 years, 1040 = 20 years
        t = range(0, 520)
        for x in t:
            # Write param names to csv/text file
            #if x == 0:
                #Write all run values to text/csv file
                    # writer.writerow([x, Y,
                    #                   B, M, Z, totalGoats,
                    #                   D, S, Mem[0] / D / inventoryGoal, W, # inventory ratio
                    #                   initialPrice, P,
                    #                   0,
                    #                   0,
                    #                   0,
                    #                   0,
                    #                   0,
                    #                   0,
                    #                   0, demandIntercept, demandSchedule,
                    #                   producerProfit,
                    #                   accProducerProfit])

            # Production - based Delta values

            # Young goats
            youngGoatsBorn = B * kiddingRate * (1-stillbirthRate)/kiddingInterval
            youngGoatsGrowingIntoBreeding = Y / ageFirstKidding * ratioFemales
            youngGoatsGrowingIntoMales =  Y / ageFirstKidding * (1 - ratioFemales)
            youngGoatsDying = Y * (youngMortality/ageFirstKidding)

            # Delta
            dYdt = youngGoatsBorn - youngGoatsDying - youngGoatsGrowingIntoBreeding - youngGoatsGrowingIntoMales

            # Cumulative
            accYoungPop = accYoungPop + youngGoatsBorn
            accYoungDeaths = accYoungDeaths + youngGoatsDying

            # Breeding goats
            breedingGoatsGrowingOld = B / avgBreedingPeriod
            breedingGoatsDying = B * breedingMortality/avgBreedingPeriod

            # Check to make sure we only take B to 0.0 and not negative...
            # if B + youngGoatsGrowingIntoBreeding- breedingGoatsGrowingOld - breedingGoatsDying < W * (minHerdSize * nFarmers):
            #     breedingGoatsSelling = (B + youngGoatsGrowingIntoBreeding - breedingGoatsGrowingOld - breedingGoatsDying - W * minHerdSize * nFarmers)
            # if breedingGoatsSelling < 0:
            #     breedingGoatsSelling = 0

            dBdt = youngGoatsGrowingIntoBreeding - breedingGoatsGrowingOld - breedingGoatsDying

            if B + dBdt <0:
                dBdt = -B

            accBreedingPop = accBreedingPop + youngGoatsGrowingIntoBreeding
            accBreedingDeaths = accBreedingDeaths + breedingGoatsDying

            # Male goats
            #youngGoatsGrowingIntoMales = Y / ageFirstKidding * (1-ratioFemales)
            maleGoatsGrowingOld = M/avgBreedingPeriod
            maleGoatsDying =  M * breedingMortality / avgBreedingPeriod

            dMdt = youngGoatsGrowingIntoMales - maleGoatsGrowingOld - maleGoatsDying

            if M + dMdt <0:
                dMdt = -M

            accMalePop = accMalePop + youngGoatsGrowingIntoMales
            accMaleDeaths = accMaleDeaths + maleGoatsDying

            # Old goats
            oldGoatsDying = Z * (oldMortality/(timeLife - avgBreedingPeriod - ageFirstKidding))
            dZdt = maleGoatsGrowingOld + breedingGoatsGrowingOld - oldGoatsDying

            if Z + dZdt < 0:
                dZdt = -Z

            # Can all demand be satisfied?
            if S >= D + refConsuming:
                pctMet = 1
            elif S < D + refConsuming:
                pctMet = S/(D + refConsuming)
            if pctMet > 1:
                pctMet =1
            if pctMet < 0:
                pctMet = 0

            # Set reference values for consumption and sales
            goatsSelling = pctMet*D
            goatsConsuming = pctMet*refConsuming

            # Assume we will meet all demands with older goats
            oldGoatsConsuming = goatsConsuming
            oldGoatsSelling = goatsSelling
            maleGoatsSelling = 0
            maleGoatsConsuming = 0
            youngGoatsSelling = 0
            youngGoatsConsuming = 0
            breedingGoatsSelling = 0
            breedingGoatsConsuming = 0


            if Z + dZdt < oldGoatsConsuming + oldGoatsSelling:
                pctCovered = (Z + dZdt)/(oldGoatsConsuming + oldGoatsSelling)
                oldGoatsConsuming = oldGoatsConsuming * pctCovered
                oldGoatsSelling = oldGoatsSelling * pctCovered
                if pctCovered < 1:
                    if M + dMdt - (goatsSelling - oldGoatsSelling + goatsConsuming - oldGoatsConsuming) >= 0:
                        maleGoatsSelling = goatsSelling - oldGoatsSelling
                        maleGoatsConsuming = goatsConsuming - oldGoatsConsuming
                    elif M + dMdt - (goatsSelling - oldGoatsSelling + goatsConsuming - oldGoatsConsuming) < 0:
                        maleGoatsSelling = goatsSelling / (goatsSelling + goatsConsuming) * (M + dMdt)
                        maleGoatsConsuming = goatsConsuming / (goatsSelling + goatsConsuming) * (M + dMdt)
                        if oldGoatsSelling + maleGoatsSelling + oldGoatsConsuming + maleGoatsConsuming < goatsSelling + goatsConsuming:
                            if Y + dYdt - (goatsSelling - oldGoatsSelling - maleGoatsSelling + goatsConsuming - oldGoatsConsuming - maleGoatsConsuming) >=0:
                                youngGoatsSelling = goatsSelling - oldGoatsSelling - maleGoatsSelling
                                youngGoatsConsuming = goatsConsuming - oldGoatsConsuming - maleGoatsConsuming
                            if Y + dYdt - (goatsSelling - oldGoatsSelling - maleGoatsSelling + goatsConsuming - oldGoatsConsuming - maleGoatsConsuming) <0:
                                youngGoatsSelling = goatsSelling / (goatsSelling + goatsConsuming)*(Y + dYdt)
                                youngGoatsConsuming = goatsConsuming / (goatsSelling + goatsConsuming)*(Y + dYdt)
                                if goatsSelling - oldGoatsSelling - maleGoatsSelling -youngGoatsSelling + goatsConsuming - oldGoatsConsuming - maleGoatsConsuming- youngGoatsConsuming >=0:
                                    if B + dBdt - (goatsSelling - oldGoatsSelling - maleGoatsSelling - youngGoatsSelling + goatsConsuming - oldGoatsConsuming - maleGoatsConsuming - youngGoatsConsuming) - minHerdSize * nFarmers >= 0:
                                        breedingGoatsSelling = goatsSelling - oldGoatsSelling - maleGoatsSelling - youngGoatsSelling
                                        breedingGoatsConsuming = goatsConsuming - oldGoatsConsuming - maleGoatsConsuming - youngGoatsConsuming
                                    elif B + dBdt - (goatsSelling - oldGoatsSelling - maleGoatsSelling - youngGoatsSelling + goatsConsuming - oldGoatsConsuming - maleGoatsConsuming - youngGoatsConsuming) - minHerdSize * nFarmers < 0:
                                        breedingGoatsSelling = goatsSelling / (goatsSelling + goatsConsuming) * (B + dBdt - minHerdSize * nFarmers)
                                        breedingGoatsConsuming = goatsConsuming / (goatsSelling + goatsConsuming) * (B + dBdt - minHerdSize * nFarmers)
                                        if breedingGoatsSelling < 0:
                                            breedingGoatsSelling = 0
                                        if breedingGoatsConsuming < 0:
                                            breedingGoatsConsuming = 0

            if W == 0:
                oldGoatsConsuming = 0
                oldGoatsSelling = 0
                maleGoatsSelling = 0
                maleGoatsConsuming = 0
                youngGoatsSelling = 0
                youngGoatsConsuming = 0
                breedingGoatsSelling = 0
                breedingGoatsConsuming = 0


            goatsSold = oldGoatsSelling + maleGoatsSelling + youngGoatsSelling + breedingGoatsSelling
            goatsConsumed = oldGoatsConsuming + maleGoatsConsuming+ youngGoatsConsuming + breedingGoatsConsuming

            producerProfit = goatsSold * (P - prodCost)
            producerCost = goatsSold * prodCost

            accProducerProfit = accProducerProfit + producerProfit
            accProducerCost = accProducerCost + producerCost

            accGoatsSold = accGoatsSold + goatsSold
            accGoatsConsumed = accGoatsConsumed + goatsConsumed
            accGoatDemand = accGoatDemand + D

            # The base exponentiated to PriceElasticityInventoryRatio (M/D/InventoryGoal) is perceived inventory coverage
            dPdt = initialPrice * np.power(((Mem[0] / D) / inventoryGoal),  priceElasticityInventoryRatio) - P
            P = P + dPdt / demandAdjustmentTime
            if P < 0:
                P = 1
            demandSchedule = np.exp(demandIntercept)*np.power(P, priceElasticityDemand)

            dDdt = (demandSchedule - D) / demandAdjustmentTime

            #  Now calculate the new values of the State Variables...
            dYdt = dYdt - youngGoatsSelling - youngGoatsConsuming
            Y = Y + dYdt  # Young goats
            if Y < 0:
                Y = 0
            #A = A + dAdt  # Adult goats
            dMdt = dMdt - maleGoatsSelling - maleGoatsConsuming
            M = M + dMdt # Male goats
            if M < 0:
                M = 0
            dBdt = dBdt - breedingGoatsSelling - breedingGoatsConsuming
            B = B + dBdt  # Breeding goats
            if B < 0:
                B = 0
            dZdt = dZdt - oldGoatsSelling - oldGoatsConsuming
            Z = Z + dZdt  # Old goats
            if Z < 0:
                Z = 0
            totalGoats = Y+B+M+Z

            # New willingness to sell
            # Willingness to sell in next time step
            if P < prodCost:
                W = 0 # Producers won't sell for a loss
            elif (P >= prodCost):
                #dWdt = (P/initialPrice - W)/demandAdjustmentTime # Simplified linear relationship
                # Logistic reponse to average herd size, linear response to Price
                dWdt = (1/(1+exp(-(-0.71888  + 0.06260 * totalGoats/nFarmers))) * (-1.342396 + 0.007361*P)/3.2125-W)
                W = W + dWdt
            if(W > 1.0):    #Check one more time to make that Willingness to Sell bounded at 1.0
                W = 1.0
            if (W < 0):
                W = 0

            # New supply
            #S = W * (M + Z + Y + B - minHerdSize*nFarmers)   # Supply for consumers to
            S = W * (M + Z + Y + B ) - minHerdSize*nFarmers
            if S < 0:
                S = 1

            # Memory
            # Aggregator memory
            Mem.pop(0) # Delete first value
            if(S<1.0):
                Mem.append(1.0)  #  Append a one so that we do not get a divide by zero
            else:
                marketPerceptErr  = 1.0 #random.uniform(0.85, 1.15)
                Mem.append(S*marketPerceptErr) # Append last value

            #demandShock = random.random() # Can be used to affect demand

            D = D + dDdt   # Customer Orders for next time step

            # Accumulated mortalities
            accYoungMortality = accYoungDeaths/accYoungPop * 100
            accBreedingMortality = accBreedingDeaths/accBreedingPop * 100
            accMaleMortality = accMaleDeaths/accMalePop * 100
            # Write timeseries to text/csv file

            #with open("vc-results.csv", 'a', newline = '') as file:

            with open("vc-results_final.txt", 'a', newline = '') as file:
            #with open("vc-results_16traj.csv", 'a', newline = '') as file:
                writer = csv.writer(file)
                writer.writerow([runno, x, Y,
                                 B, M, Z, totalGoats,
                                 D, S, Mem[0] / D / inventoryGoal, W, # inventory ratio
                                 initialPrice, P,
                                 goatsSold,
                                 goatsConsumed,
                                 dYdt,
                                 dBdt,
                                 dMdt,
                                 dZdt,
                                 dDdt, demandIntercept, demandSchedule,
                                 producerProfit,
                                 accProducerProfit])
                # Write outcomes of interest values to text for GSA
            if x == max(t):
                # Age distribution of goat population - to compare against literature
                pctYoungGoats = (Y / totalGoats) * 100
                pctMaleGoats = (M / totalGoats) * 100
                pctBreedingGoats = (B / totalGoats) * 100
                pctOldGoats = (Z/totalGoats) * 100
                pctChangePrice = ((P-initialPrice)/initialPrice) * 100
                pctDemandMet = accGoatsSold/accGoatDemand * 100
                pctConsumptionMet = accGoatsConsumed/(refConsuming*(max(t) + 1))

                avgWeeklyGoatsSold = accGoatsSold/(max(t) + 1)

                with open('output1_final.txt', 'a', newline='') as f:
                #with open('output1.txt', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([runno, # run number
                        round(totalGoats, 1),
                        #round(pctYoungGoats, 1),# round(pctAdultGoats, 1),
                        #round(pctBreedingGoats, 1),
                        #round(pctOldGoats, 1),
                        #round((goatsPerFarmer * nFarmers - totalGoats)/goatsPerFarmer * nFarmers * 100, 1), # Change in total goat pop
                        #round(accGoatsConsumed, 1),
                        #round(accGoatsSold, 1),
                        #round(accGoatDemand, 1),
                        round(accProducerProfit/nFarmers,1), #round(accProducerCost/nFarmers, 1),
                        round(W,2), # willingness to sell goats at end of simulation
                        round(P,2),round(S,2),round(D,2),# price, supply, demand
                        #round(totalGoats/nFarmers, 1), # avg goats per farmer at end of simulation
                        #round(accYoungMortality, 1), # accumulated young mortality
                        #round(accBreedingMortality, 1), # accumulated breeding mortality
                        round(pctChangePrice, 1),
                        round(pctDemandMet, 1),
                        round(avgWeeklyGoatsSold, 1)])

                #with open('mcf_outputs.txt', 'a', newline='') as fi:
                with open('mcf_outputs_final.txt', 'a', newline='') as fi:
                    writer = csv.writer(fi)
                    writer.writerow([runno, # run number
                                     round(totalGoats, 1),
                                     round(pctYoungGoats, 1), round(pctMaleGoats, 1),
                                     round(pctBreedingGoats, 1),
                                     round(pctOldGoats, 1),
                                     round((goatsPerFarmer * nFarmers - totalGoats)/goatsPerFarmer * nFarmers * 100, 1), # Change in total goat pop
                                     round(accGoatsConsumed, 1),
                                     round(accGoatsSold, 1),
                                     round(accGoatDemand, 1),
                                     round(accProducerProfit/nFarmers,1), round(accProducerCost/nFarmers, 1),
                                     round(W,2), # willingness to sell goats at end of simulation
                                     round(P,2),round(S,2),round(D,2),# price, supply, demand
                                     round(totalGoats/nFarmers, 1), # avg goats per farmer at end of simulation
                                     round(accYoungMortality, 1), # accumulated young mortality
                                     round(accBreedingMortality, 1), # accumulated breeding mortality
                                     round(pctChangePrice, 1),
                                     round(pctDemandMet, 1),
                                     round(avgWeeklyGoatsSold, 1)])


print("Done!")