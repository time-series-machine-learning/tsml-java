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

package experiments;

import evaluation.MultipleClassifierEvaluation;
import java.util.Arrays;
import java.util.Random;
import net.sourceforge.sizeof.SizeOf;
import utilities.ClassifierTools;
import vector_classifiers.CAWPE;
import weka.core.Instances;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class jameslTemp {
    public static void main(String[] args) throws Exception {
//        String[] args2 = {
//            "-dp=Z:/Data/UCIDelgado/",
//            "-rp=C:/Temp/jcommandertests/",
//            "-cn=C45",
//            "-dn=balloons",
//            "-f=16",
//            
//            "-sc=true",
////            "-tb=true",
//            "-d=true",
//        };
//        Experiments.main(args2);
//        
//        mbenchmark();
        
//        Foo foo = new Foo();
//        
//        System.out.println(SizeOf.sizeOf(foo));
//        System.out.println(SizeOf.deepSizeOf(foo));




//        shapeletStuff();
        t();
    }
    
    public static class Foo { 
        
    }
    
    /**
     * Repeatedly performing a sort operation on an array of ints to a) average over 
     * randomness in the initial states of the arrays, b) average over java's cooky 
     * object initialisation overhead, i.e. later iterations of the same loop 
     * always seem faster than the first (few).
     */
    public static long benchmark() { 
        long startTime = System.nanoTime();
        int n = 10000;
        int[] a = new int[n];
        Random rng = new Random(0);
        for (int j = 0; j < n; j++)
            a[j] = rng.nextInt();

        Arrays.sort(a);
        return System.nanoTime() - startTime;
    }
    
    public static void mbenchmark() { 
        int R = 1000;
        long[] times = new long[R];
        long total = 0L;
        for (int i = 0; i < R; i++) {
            times[i] = benchmark();
            total+=times[i];
//            System.out.print(times[i] + ", ");
        }
//        System.out.println("");
        
        
        long mean = 0L, max = Long.MIN_VALUE, min = Long.MAX_VALUE;
        for (long time : times) { 
            mean += time;
            if (time < min)
                min = time;
            if (time > max)
                max = time;
        }
        mean/=R;
        
        int halfR = R/2;
        long median = R % 2 == 0 ? 
                (times[halfR] + times[halfR+1]) / 2 :
                times[halfR];
        
        System.out.println("NANOS ---");
        System.out.println("total = " + total);
        System.out.println("min = " + min);
        System.out.println("max = " + max);
        System.out.println("mean = " + mean);
        System.out.println("median = " + median);
        
        double d = 1000000000;
        System.out.println("SECONDS ---");
        System.out.println("total = " + total/d);
        System.out.println("min = " + min/d);
        System.out.println("max = " + max/d);
        System.out.println("mean = " + mean/d);
        System.out.println("median = " + median/d);
    }
    
    
    
    
    
    
    
    
    
    public static void t() throws Exception {
        String [] baseClassifiers = new String[] { "NN", "NB", "SVML", "SVMQ", "RandF", "RotF", "C45", "bayesNet", "CAWPE" };
        new MultipleClassifierEvaluation("C:/Temp/ShapeletQuery/", "ANALYSIS", 30).
            setTestResultsOnly(false).
//            setBuildMatlabDiagrams(true).
            setBuildMatlabDiagrams(false).
            setDatasets(new String[] { "Wine" }).
            readInClassifiers(baseClassifiers, baseClassifiers, "C:/Temp/ShapeletQuery/").
            runComparison();
    }
    
    public static void shapeletStuff() throws Exception { 
        
        
        String dataPath = "Z:/Data/Shapelets30Resamples/";
        String dataset = "Wine";
        String[] dtemp = { dataset };
        int numFolds = 30;
        
        String baseResPath = "C:/Temp/ShapeletQuery/";
        
        
        Experiments.ExperimentalArguments exp = new Experiments.ExperimentalArguments();
        exp.generateErrorEstimateOnTrainSet = true;
        exp.dataReadLocation = dataPath;
        exp.resultsWriteLocation = baseResPath;
        
        String [] baseClassifiers = new String[] { "NN", "NB", "SVML", "SVMQ", "RandF", "RotF", "C45", "bayesNet" };
        Experiments.setupAndRunMultipleExperimentsThreaded(exp, baseClassifiers, dtemp, 0, numFolds);
                
        for (int fold = 5; fold < numFolds; fold++) {
            
            Instances train = ClassifierTools.loadData(dataPath + dataset + "/" + dataset + fold + "_TRAIN.arff");
            Instances test = ClassifierTools.loadData(dataPath + dataset + "/" + dataset + fold + "_TEST.arff");
            
            CAWPE c = new CAWPE();
            c.setOriginalCAWPESettings();
            
            c.setResultsFileLocationParameters(baseResPath, dataset, fold);
            c.setBuildIndividualsFromResultsFiles(true);
            c.setRandSeed(fold);
            c.setPerformCV(true);

            //'custom' classifier built, now put it back in the normal experiments pipeline
            exp.classifierName = "CAWPE";
            exp.foldId = fold;
            
            String fullResPath = baseResPath + "CAWPE" + "/Predictions/" + dataset + "/";
            Experiments.runExperiment(exp,train,test,c,fullResPath);
            
        }
        
    }
    
    
    
}
