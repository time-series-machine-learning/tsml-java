/*
 * Copyright (C) 2019 xmw13bzu
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package intervals;

import evaluation.storage.ClassifierResults;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class IntervalAnalysis {

    public static void main(String[] args) {
        
    }
    
    
    public static void postprocessBestInterval_Oracle() { 
        //for each exp, select the single best interval based on test error and test with that
        //question: is there potentially a benefit to be gained through interval section
        //sub questions: 
        //      - what datasets have the biggest potential gain? can these be grouped at all? 
        //      - how many datasets see no gain, i.e. best interval is the full series or close to it?
        //      - are the best intervals consistent across different classifiers? 
        //      - what kind of reduction in series lengths and thus final prediction time can be acheived, if the interval search is ignored? 
        
        //todo
    }
    
    public static void postprocessBestInterval_TrainEstimate() { 
        //for each exp, select the single best interval based on train error and test with that
        //question: if interval selection is shown to be useful above, can it be found in a naive/almost brute force setup on the train data? 
        //sub questions: 
        //      - what is the average difference between potential (oracle) and realised (selected from train) performance gain? 
        //      - does accuracy/performance DECREASE for any dataset/classifier, i.e. it erroneously tries to select a sub-interval? 
        //      - 
        
        //todo
    }
    
    public static void postprocessBestInterval_CheapEstimatingExpensive() { 
        //for each exp, select the single best interval based on train error WITH A CHEAP CLASSIFER (e.g. ed, svml etc) 
        //        and test WITH AN EXPENSIVE CLASSIFIER (e.g. rotf, tsf hive-cote) with that interval
        //question: do the intervals selected by cheap classifiers improve performance for more expensive classifiers?
        //question: is the time to estimate the intervals with the cheap classifier and then build and test on the single interval
        //      shorter than the time taken to build the expensive classifier on the full data straight up? 
        
        //todo
    }
    
    public static void example_gunpointIntervals() throws Exception { 
        //take gunpoint in particular, a well-studied dataset where the useful intervals have been documented before
        //do the different interval test accuracies reflect the expected interval importances based on other works? 
        //how does the single 'oracle' interval correspond to the above? 
        //how does the selected interval from train data correspond to the above? 
        //draw some sort of heatmap figure relaying this info for paper
        
        String baseWritePath = "E:/Intervals/GunpointExampleAna/";
        
        String baseResPath = "";
        String dataset = "Gunpoint";
        String baseClassifier = "ED"; 
        int fold = 0;
        
        ClassifierResults[] allIntervalsRes = new ClassifierResults[IntervalHeirarchy.maxNumDifferentIntervals];
        
        for (int i = 0; i < IntervalHeirarchy.maxNumDifferentIntervals; i++) {
            String cname = IntervalHeirarchy.buildIntervalClassifierName(baseClassifier, IntervalHeirarchy.defineInterval(i));
            allIntervalsRes[i] = new ClassifierResults(baseResPath + cname + "/Predictions/" + dataset + "/testFold" + fold + ".csv");
        }
        
    }
    
    public static void example_AlcoholSpectra() { 
        //take the ethanol concentration and jwlabel problems in particular. we know that the information lies in different intervals for each
        //perform same analysis as for gun point
        //do the intervals align with the expect ones for each dataset?
        //does performance improve? 
        //compare e.g ed, ed_interval, to pls and pls_i as benchmarks
        
        //todo
    }
}
