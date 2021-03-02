/*
 * Copyright (C) 2019 xmw13bzu
 *
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */

package examples;

import experiments.data.DatasetLoading;
import utilities.InstanceTools;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Examples to show different ways of loading and basic handling of datasets
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class DataHandling {

    public static void main(String[] args) throws Exception {
        
        // We'll be loading the ItalyPowerDemand dataset which is distributed with this codebase
        String basePath = "src/main/java/experiments/data/tsc/";
        String dataset = "ItalyPowerDemand";
        int seed = 1;
        
        Instances train;
        Instances test;
        Instances[] trainTest;
        
        
        
        
        ///////////// Loading method 1: loading individual files
        // DatasetLoading.loadData...(...)
        // For loading in a single arff without performing any kind of sampling. Class value is 
        // assumed to be the last attribute
        
        train = DatasetLoading.loadDataThrowable(basePath + dataset + "/" + dataset + "_TRAIN.arff");
        test = DatasetLoading.loadDataThrowable(basePath + dataset + "/" + dataset + "_TEST.arff");
        
        // We could then resample these, while maintaining train/test distributions, using this
        
        trainTest = InstanceTools.resampleTrainAndTestInstances(train, test, 1);
        train = trainTest[0];
        test = trainTest[1];
        
        
        
        
        
        
        ///////////// Loading method 2: sampling directly
        // DatasetLoading.sampleDataset(...)
        // Wraps the data loading and sampling performed above. Read in a dataset either
        // from a single complete file (e.g. uci data) or a predefined split (e.g. ucr/tsc data) 
        // and resamples it according to the seed given. If the resampled fold can already 
        // be found in the read location ({dsetname}{foldid}_TRAIN and _TEST) then it will
        // load those. See the sampleDataset(...) javadoc
        
        trainTest = DatasetLoading.sampleDataset(basePath, dataset, seed);
        train = trainTest[0];
        test = trainTest[1];
        
        
        
        
        
        
        ///////////// Loading method 3: sampling the built in dataset
        // DatasetLoading.sampleDataset(...)
        // Because ItalyPowerDemand is distributed with the codebase, there's a wrapper 
        // to sample it directly for quick testing 
        
        trainTest = DatasetLoading.sampleItalyPowerDemand(seed);
        train = trainTest[0];
        test = trainTest[1];
        
        
        
        
        
        
        //////////// Data inspection and handling:
        // We can look at the basic meta info
        
        System.out.println("train.relationName() = " + train.relationName());
        System.out.println("train.numInstances() = " + train.numInstances());
        System.out.println("train.numAttributes() = " + train.numAttributes());
        System.out.println("train.numClasses() = " + train.numClasses());
        
        // And the individual instances
        
        for (Instance inst : train)
            System.out.print(inst.classValue() + ", ");
        System.out.println("");
        
        
        
        
        
        
        
        
        
        // Often for speed we just want the data in a primitive array
        // We can go to and from them using this sort of procedure
        
        // Lets keeps the class labels separate in this example
        double[] classLabels = train.attributeToDoubleArray(train.classIndex()); // aka y_train
        
        boolean removeLastVal = true;
        double[][] data = InstanceTools.fromWekaInstancesArray(train, removeLastVal); // aka X_train
        
        // We can then do whatever fast array-optimised stuff, and shove it back into an instances object
        Instances reformedTrain = InstanceTools.toWekaInstances(data, classLabels);
    }
    
}
