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

import experiments.Experiments.ExperimentalArguments;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import timeseriesweka.filters.NormalizeCase;
import utilities.FileHandlingTools;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class IntervalExperiments {

    public static final int maxNumIntervalPoints = 20; //so 21 values really, 0 .. 20 corresponding to props 0 .. 1 
    public static final int maxNumIntervals = 230; 
    
    public static void zipTest() throws IOException {
        String dirToZip = "C:/Temp/intervalExpTest/ED_0.00_0.05/Predictions/ACSF1/";
        String zipName = "C:/Temp/intervalExpTest/ED_0.00_0.05/Predictions/ACSF1folds.zip";
        
        zip(dirToZip, zipName);
        deleteContents(new File(dirToZip));
        move(zipName, dirToZip);
    }
    
    public static void deleteContents(File dir) throws IOException {
        for (File file: dir.listFiles()) {
            if (file.isDirectory())
                deleteContents(file);
            file.delete();
        }
    }
    
    //https://stackoverflow.com/a/32052016
    public static void zip(String sourceDirPath, String zipFilePath) throws IOException {
        Path p = Files.createFile(Paths.get(zipFilePath));
        try (ZipOutputStream zs = new ZipOutputStream(Files.newOutputStream(p))) {
            Path pp = Paths.get(sourceDirPath);
            Files.walk(pp)
                .filter(path -> !Files.isDirectory(path))
                .forEach(path -> {
                    ZipEntry zipEntry = new ZipEntry(pp.relativize(path).toString());
                    try {
                      zs.putNextEntry(zipEntry);
                      Files.copy(path, zs);
                      zs.closeEntry();
                    } catch (IOException e) {
                        System.err.println(e);
                    }
                });
        }
    }
    
    public static void move(String fullFileToMove, String dirToMoveTo) {
        try{
    	   File f =new File(fullFileToMove);
    	   if(!f.renameTo(new File(dirToMoveTo + f.getName())))
    		throw new Exception("zip moving failed, unspecified reason");
    	}catch(Exception e){
            System.out.println(e);
    	}
    }
    
    public static void main(String[] args) throws Exception {
//        zipTest();
//        localExps(args);
        
//        args = new String[] { "0", "-dp=Z:/Data/TSCProblems2018_Folds/", "-rp=C:/Temp/intervalExpTest/", "-cn=ED" };
        clusterExps(args);
    }
    
    /**
     * @param args [ intervalID, dataPath, resPath, classifier ] 
     * 
     * will run all datasets (tsc128) and folds (10) of given interval/classifier,
     * zip the results and delete the original files
     */
    public static void clusterExps(String[] args) throws Exception {
        int folds = 30;
        String[] dsets = DataSets.tscProblems2018;
//        dsets = Arrays.copyOfRange(dsets, 0, 5);
        
        String classifier = null;
        
        for (String dset : dsets) {
            for (int f = 1; f <= folds; f++) {
                String[] newArgs=new String[9];
                newArgs[0]=args[0];
                newArgs[1]=args[1];
                newArgs[2]=args[2];//Where to get data                
                newArgs[3]=args[3];//Where to write results                
                newArgs[4]="-gtf=true"; //Whether to generate train files or not               
                newArgs[5]=args[4]; //Classifier name
                newArgs[6]="-dn="+dset; //Problem file   
                newArgs[7]="-f="+f;//Fold number (fold number 1 is stored as testFold0.csv, its a cluster thing)   
                newArgs[8]="-tb=true";

                classifier = runExperiment(newArgs);
            }
        }
        
        String baseDir = args[2].split("=")[1];
        
        String dirToZip = baseDir + classifier + "/";
        String zipName = baseDir + classifier + ".zip";
        
        zip(dirToZip, zipName);
        deleteContents(new File(dirToZip));
        move(zipName, dirToZip);
    }
    
    public static void localExps(String[] args) throws Exception {
        
        boolean norm = false;
        
        for (String dset : DataSets.tscProblems2018) {
            for (int f = 1; f <= 10; f++) {
                for (int i = 227; i < maxNumIntervals; i++) {
                    args=new String[9];
                    args[0]=""+norm;
                    args[1]=""+i;
                    args[2]="-dp=Z:/Data/TSCProblems2018_Folds/";//Where to get data                
                    args[3]="-rp=C:/Temp/intervalExpTest/";//Where to write results                
                    args[4]="-gtf=true"; //Whether to generate train files or not               
                    args[5]="-cn=ED"; //Classifier name
                    args[6]="-dn="+dset; //Problem file   
                    args[7]="-f="+f;//Fold number (fold number 1 is stored as testFold0.csv, its a cluster thing)   
                    args[8]="-tb=true";

                    runExperiment(args);
                }
            }
        }
    }
    
    public static String runExperiment(String[] args) throws Exception {
        
        boolean normaliseInterval = Boolean.parseBoolean(args[0]);
        int intervalID = Integer.parseInt(args[1]);
        double[] interval = defineInterval(intervalID);
        args = Arrays.copyOfRange(args, 2, args.length);
        
        //semi-manual experiment setup to get cawpe to write it's individuals predictions
        ExperimentalArguments exp = new ExperimentalArguments(args);
        System.out.println(exp.toShortString());
       
        Classifier classifier = ClassifierLists.setClassifier(exp);
        exp.classifierName = exp.classifierName + "_" + String.format("%.2f", interval[0]).split("\\.")[1] 
                                                + "_" + String.format("%.2f", interval[1]).split("\\.")[1]; 
        
        String fullWriteLoc = exp.resultsWriteLocation + exp.classifierName + "/Predictions/" + exp.datasetName + "/";
        (new File(fullWriteLoc)).mkdirs();
        String fullTargetFile = fullWriteLoc + "testFold" + exp.foldId + ".csv";
        
        if (experiments.CollateResults.validateSingleFoldFile(fullTargetFile))
            System.out.println(exp.toShortString() + " already exists at "+fullTargetFile+", exiting.");
        else {
            Instances[] data = Experiments.sampleDataset(exp.dataReadLocation, exp.datasetName, exp.foldId);
            data[0] = crop_proportional(data[0], interval[0], interval[1], normaliseInterval);
            data[1] = crop_proportional(data[1], interval[0], interval[1], normaliseInterval);

            Experiments.runExperiment(exp, data[0], data[1], classifier, fullWriteLoc);
        }
        
        return exp.classifierName;
    }
    
    public static double[] defineInterval(int intID) throws Exception {
        int startId = 0;
        int endId = 1;

        int c = 0;        
        while (c != intID) { 
            if (endId++ > maxNumIntervalPoints) {
                startId++;
                endId = startId + 1;
                
                if (startId > maxNumIntervalPoints-1)
                    throw new Exception("something wrong in interval defintion, startId=" + startId + " endId=" + endId + " intId=" + intID);
            }
            
            c++;
        }
        
        
        double startProp = (double)startId / maxNumIntervalPoints;
        double endProp = (double)endId / maxNumIntervalPoints;
        
        return new double[] { startProp, endProp };
    }
    
    //maybe theres a filter that does what i want?... faster to just do it here
    public static Instances crop_proportional(Instances insts, double startProp, double endProp, boolean normalise) throws Exception { 
//        System.out.println(insts.numAttributes());
//        System.out.println(insts.numInstances());
//        System.out.println(insts.numClasses());

        int startTimePoint = (int) ((insts.numAttributes()-1) * startProp);
        int endTimePoint = (int) ((insts.numAttributes()-1) * endProp);
        
//        System.out.println("");
//        System.out.println("inteveral = " + startProp + " - " + endProp);
//        System.out.println("startTimePoint = " + startTimePoint);
//        System.out.println("endTimePoint = " + endTimePoint);
//        System.out.println("");
        
        if (startTimePoint == endTimePoint) {//pathological case for very short series
            if (endTimePoint < (insts.numAttributes()-2))
                endTimePoint++;
            else if (startTimePoint > 0)
                startTimePoint--;
            else 
                throw new Exception("Interval wont work, " + insts.relationName() + " start=" + startProp + " end=" + endProp + " numAtts=" + insts.numAttributes());
        }
        
        Instances cropped = new Instances(insts);

        for (int ind = cropped.numAttributes()-2; ind >= 0; ind--)
            if (ind < startTimePoint || ind > endTimePoint)
                cropped.deleteAttributeAt(ind);
        
//        System.out.println(cropped.numAttributes());
//        System.out.println(cropped.numInstances());
//        System.out.println(cropped.numClasses());
        
        if (normalise)
            return (new NormalizeCase()).process(cropped);
        else
            return cropped;
    }
    
}

