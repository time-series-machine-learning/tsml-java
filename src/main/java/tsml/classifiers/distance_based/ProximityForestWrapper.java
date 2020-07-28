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

package tsml.classifiers.distance_based;

import core.AppContext;
import core.contracts.Dataset;
import datasets.ListDataset;
import evaluation.MultipleClassifierEvaluation;
import experiments.Experiments;
import java.util.Random;
import trees.ProximityForest;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;

/**
 * 
 * An in-progress wrapper/conversion class for the java Proximity Forest implementation.
 * 
 * The code as-is on the github page does not return distributions when predicting. Therefore
 * in our version of the jar I have added a predict_proba() to be used here instead of predict().
 * Existing proximity code is UNEDITED, and predict_proba simply returns the distribution of 
 * the num_votes class member instead of resolving ties internally and returning the single,
 * majority voted for, class value. As such, metrics that do not consider probabilistic performance 
 * (accuracy, most importantly) should be identical when training and testing on identical data
 * with identical seeds. The only part where this falls down is in tie resolution for 
 * the majority class, todo. 
 * 
 * tl;dr, performance on average should be significantly the same towards vanishing p values
 * 
 * NOTE1: we are by-passing the test method which is ultimately foreach inst { predict() },
 * and so proximity forest's internal results object is empty. This has no other sides effects for 
 * our purposes here.
 * 
 * NOTE2: because of the static AppContext holding e.g random seeds etc, do not run multiple 
 * experiments using ProximityForest in parallel, i.e with Experiments.setupAndRunMultipleExperimentsThreaded(...)
 * 
 * TODO: weka/tsc interface implementations etc, currently this is simply in a runnable state 
 * for basic experimental runs to compare against. Need: TechnicalInformationHandler, need to do the get/setOptions, 
 * could do parameter searches if wanted, etcetc.
 * 
 * 
 * Github code:   https://github.com/fpetitjean/ProximityForestWeka
 * 
 * @article{DBLP:journals/corr/abs-1808-10594,
 *   author    = {Benjamin Lucas and
 *                Ahmed Shifaz and
 *                Charlotte Pelletier and
 *                Lachlan O'Neill and
 *                Nayyar A. Zaidi and
 *                Bart Goethals and
 *                Fran{\c{c}}ois Petitjean and
 *                Geoffrey I. Webb},
 *   title     = {Proximity Forest: An effective and scalable distance-based classifier
 *                for time series},
 *   journal   = {CoRR},
 *   volume    = {abs/1808.10594},
 *   year      = {2018},
 *   url       = {http://arxiv.org/abs/1808.10594},
 *   archivePrefix = {arXiv},
 *   eprint    = {1808.10594},
 *   timestamp = {Mon, 03 Sep 2018 13:36:40 +0200},
 *   biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1808-10594},
 *   bibsource = {dblp computer science bibliography, https://dblp.org}
 * }
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class ProximityForestWrapper extends AbstractClassifier implements Randomizable {

    //from paper, pg18-19: 
    /*
        4.2 Experiments on the UCR Archive
    
        ...
    
        The Proximity Forest results are obtained for 100 trees
        with selection between 5 candidates per node. A detailed discussion about the
        Proximity Forest parameters will be performed in Section 4.2
    */
    private int num_trees = 100;                
    private int num_candidates_per_split = 5;   
    private boolean random_dm_per_node = true;  
    
    public ProximityForest pf;
    
    private int numClasses;
    private Instances header;
    
    public ProximityForestWrapper() {
    }

    public int getNum_trees() {
        return num_trees;
    }

    public void setNum_trees(int num_trees) {
        this.num_trees = num_trees;
    }

    public int getNum_candidates_per_split() {
        return num_candidates_per_split;
    }

    public void setNum_candidates_per_split(int num_candidates_per_split) {
        this.num_candidates_per_split = num_candidates_per_split;
    }

    public boolean isRandom_dm_per_node() {
        return random_dm_per_node;
    }

    public void setRandom_dm_per_node(boolean random_dm_per_node) {
        this.random_dm_per_node = random_dm_per_node;
    }
        
    public void setSeed(int seed) { 
        AppContext.rand_seed = seed;
        AppContext.rand = new Random(seed);        
    }
    
    public int getSeed() { 
        return (int)AppContext.rand_seed;
    }
    
    private Dataset toPFDataset(Instances insts) {
        Dataset dset = new ListDataset(insts.numInstances());
        
        for (Instance inst : insts)
            dset.add((int)inst.classValue(), getSeries(inst));
        
        return dset;
    }
    
    private double[] getSeries(Instance inst) {
        double[] d = new double[inst.numAttributes()-1];
        for (int i = 0; i < d.length; i++)
            d[i] = inst.value(i);
        return d;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        //init
        numClasses = data.numClasses();
        header = new Instances(data,0);

        if (AppContext.rand == null) AppContext.rand = new Random(AppContext.rand_seed);
                
        AppContext.num_trees = num_trees;
        AppContext.num_candidates_per_split = num_candidates_per_split;
        AppContext.random_dm_per_node = random_dm_per_node;
        
        pf = new ProximityForest((int) AppContext.rand_seed); //just an id 
        
        
        //actual work
        Dataset pfdata = toPFDataset(data);
        pf.train(pfdata);
    }
    
    @Override
    public double[] distributionForInstance(Instance inst) throws Exception {
//        header.add(inst);
//        Dataset dset = toPFDataset(header);
//        header.remove(0);
//        
//        double[] dist = new double[inst.numClasses()]; 
//        ProximityForestResult pfres = pf.test(dset);
//        
        
        return pf.predict_proba(getSeries(inst), numClasses);
    }


    @Override
    public double classifyInstance(Instance inst) throws Exception {
        double[] probs = distributionForInstance(inst);

        int maxClass = 0;
        for (int n = 1; n < probs.length; ++n) {
            if (probs[n] > probs[maxClass]) {
                maxClass = n;
            }
            else if (probs[n] == probs[maxClass]){
                if (AppContext.rand.nextBoolean()){
                    maxClass = n;
                }
            }
        }

        return maxClass;
    }
    
    public static void main(String[] args) throws Exception {
        
//        ProximityForestWrapper pf = new ProximityForestWrapper();
//        pf.setSeed(0);
//        System.out.println(ClassifierTools.testUtils_getIPDAcc(pf));
//        System.out.println(ClassifierTools.testUtils_confirmIPDReproduction(pf, 0.966958211856171, "2019_09_26"));
        
        Experiments.ExperimentalArguments exp = new Experiments.ExperimentalArguments();

        exp.dataReadLocation = "Z:/Data/TSCProblems2015/";
        exp.resultsWriteLocation = "C:/Temp/ProximityForestWekaTest/";
        exp.classifierName = "ProximityForest";
//        exp.datasetName = "BeetleFly";
//        exp.foldId = 0;
//        Experiments.setupAndRunExperiment(exp);


        
        String[] classifiers = { "ProximityForest" };
        
        String[] datasets = {
            "Beef", // 30,30,470,5
            "Car", // 60,60,577,4
            "Coffee", // 28,28,286,2
            "CricketX", // 390,390,300,12
            "CricketY", // 390,390,300,12
            "CricketZ", // 390,390,300,12
            "DiatomSizeReduction", // 16,306,345,4
            "fish", // 175,175,463,7
            "GunPoint", // 50,150,150,2
            "ItalyPowerDemand", // 67,1029,24,2
            "MoteStrain", // 20,1252,84,2
            "OliveOil", // 30,30,570,4
            "Plane", // 105,105,144,7
            "SonyAIBORobotSurface1", // 20,601,70,2
            "SonyAIBORobotSurface2", // 27,953,65,2
            "SyntheticControl", // 300,300,60,6
            "Trace", // 100,100,275,4
            "TwoLeadECG", // 23,1139,82,2  
        };
        int numFolds = 30;

        
        //Because of the static app context, best not run multithreaded, stick to single threaded
        for (String dataset : datasets) {
            for (int f = 0; f < numFolds; f++) {
                exp.datasetName = dataset;
                exp.foldId = f;
                Experiments.setupAndRunExperiment(exp);
            }
        }
        
        
        MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation(exp.resultsWriteLocation +"ANA/", "sanityCheck", numFolds);
        mce.setBuildMatlabDiagrams(false);
        mce.setTestResultsOnly(true);
        mce.setDatasets(datasets);
        mce.readInClassifier(exp.classifierName, exp.resultsWriteLocation);
//        mce.readInClassifier("DTWCV", "Z:/Results_7_2_19/FinalisedRepo/"); //no probs, leaving it 
        mce.readInClassifier("RotF", "Z:/Results_7_2_19/FinalisedRepo/");
        mce.runComparison();
    }
}
