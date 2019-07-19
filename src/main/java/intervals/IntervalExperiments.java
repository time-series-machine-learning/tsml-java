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

import experiments.ClassifierLists;
import experiments.data.DatasetLists;
import experiments.Experiments;
import experiments.Experiments.ExperimentalArguments;
import experiments.data.DatasetLoading;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
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
    
    //https://www.admfactory.com/how-to-decompress-files-from-zip-format-using-java/
    public static void unzip(File zipFile, File output) {

	byte[] buffer = new byte[1024];

	try {
	    if (!output.exists())
		output.mkdirs();

	    /** get the zip file content */
	    ZipInputStream zis = new ZipInputStream(new FileInputStream(zipFile));
	    
	    /** get the first zip file entry */
	    ZipEntry ze = zis.getNextEntry();

	    while (ze != null) {

		String fileName = ze.getName();
		File newFile = new File(output + File.separator + fileName);

		System.out.println("file unzip: " + newFile.getAbsolutePath());

		/** create all non exists parent folders */
		newFile.getParentFile().mkdirs();

		FileOutputStream fos = new FileOutputStream(newFile);

		int len;
		while ((len = zis.read(buffer)) > 0)
		    fos.write(buffer, 0, len);

		fos.close();
		
		/** get the next zip file entry */
		ze = zis.getNextEntry();
	    }

	    zis.closeEntry();
	    zis.close();
	    System.out.println();
	    System.out.println("Done!");

	} catch (IOException ex) {
	    ex.printStackTrace();
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
    
    public static void unzipIntervalResultsFromCluster(String basePath) { 
        
        for (File dir : FileHandlingTools.listDirectories(basePath)) {
            File zip = new File(dir.getAbsoluteFile() + "/" + dir.getName() + ".zip");
            File unzippedContents = new File(dir.getAbsoluteFile() + "/Predictions");
            
            if (zip.exists() && !unzippedContents.exists())
                unzip(zip, dir);
        }
        
    }   
    
    private static boolean deepContains(String[] args, String find) { 
        for (String arg : args)
            if (arg.contains(find))
                return true;
        return false;
    }
    
    public static void main(String[] args) throws Exception {
//        String t = "E:/Intervals/BruteResults/Unnormed/";
//        unzipIntervalResultsFromCluster(t);
        
//        zipTest();
//        localExps(args);
//        localGunPointExps(args);
        
//        for (int i = 0; i < 250; i++) {
//            System.out.println(Arrays.toString(defineInterval(i)));
//        }

//        args = new String[] { "true", "1", "-dp=Z:/Data/TSCProblems2018_Folds/", "-rp=C:/Temp/intervalExpTest/", "-cn=ED" };

        if (deepContains(args, "-dn")) 
            manyClusterExps(args); //individual dset/fold given
        else 
            runExperiment(args); //loops over dsets/folds
        
    }
 
    public static void localGunPointExps(String[] args) throws Exception {
        for (int i = 1; i <= IntervalHeirarchy.maxNumDifferentIntervals; i++) {
            String[] newArgs=new String[9];
            newArgs[0]="false";
            newArgs[1]=i+"";
            newArgs[2]="-dp=Z:/Data/TSCProblems2018_Folds/";//Where to get data                
            newArgs[3]="-rp=C:/Temp/intervalExpTest/";//Where to write results                
            newArgs[4]="-gtf=true"; //Whether to generate train files or not               
            newArgs[5]="-cn=ED"; //Classifier name
            newArgs[6]="-dn=GunPoint"; //Problem file   
            newArgs[7]="-f=1";//Fold number (fold number 1 is stored as testFold0.csv, its a cluster thing)   
            newArgs[8]="-tb=true";

            runExperiment(newArgs);
        }
    }
    
    /**
     * @param args [ normaliseInterval?, intervalID, dataPath, resPath, classifier ] 
     * 
     * will run all datasets (tsc128) and folds (10) of given interval/classifier,
     * zip the results and delete the original files
     */
    public static void manyClusterExps(String[] args) throws Exception {
        int folds = 10;
//        String[] dsets = { "BeetleFly" };
//        String[] dsets = DatasetLists.tscProblems2018;
        String[] dsets = IntervalClassifierLists.datasets_SeriesLengthAtLeast100;
//        dsets = Arrays.copyOfRange(dsets, 0, 5);
        
        String classifier = null;
        
        for (int f = 1; f <= folds; f++) {
            for (String dset : dsets) {
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
                
                if (classifier.equals("zipped")) {
                    System.out.println("Zip exists, exiting");
                    return;
                }
            }
        }
        
        String baseDir = args[3].split("=")[1];
        
        String dirToZip = baseDir + classifier + "/";
        String zipName = baseDir + classifier + ".zip";
        
        System.out.println("All files written");
        zip(dirToZip, zipName);
        System.out.println("zipped");
        deleteContents(new File(dirToZip));
        System.out.println("orig deleted");
        move(zipName, dirToZip);
        System.out.println("zip moved");
        System.out.println("end");
    }
    
    public static void localExps(String[] args) throws Exception {
        
        boolean norm = false;
        
        for (String dset : DatasetLists.tscProblems2018) {
            for (int f = 1; f <= 10; f++) {
                for (int i = 227; i < IntervalHeirarchy.maxNumDifferentIntervals; i++) {
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
        int intervalID = Integer.parseInt(args[1]) - 1;
        double[] interval = IntervalHeirarchy.defineInterval(intervalID);
        args = Arrays.copyOfRange(args, 2, args.length);
        
        //semi-manual experiment setup to get cawpe to write it's individuals predictions
        ExperimentalArguments exp = new ExperimentalArguments(args);
        System.out.println(exp.toShortString());
       
        Classifier classifier = ClassifierLists.setClassifier(exp);
        exp.classifierName = IntervalHeirarchy.buildIntervalClassifierName(exp.classifierName, interval);
        
        
        if (new File(exp.resultsWriteLocation + exp.classifierName + "/" + exp.classifierName + ".zip").exists()) {
            System.out.println("All resutls zip exists, exiting");
            return "zipped";
        }
        
        String fullWriteLoc = exp.resultsWriteLocation + exp.classifierName + "/Predictions/" + exp.datasetName + "/";
        (new File(fullWriteLoc)).mkdirs();
        String fullTargetFile = fullWriteLoc + "testFold" + exp.foldId + ".csv";
        
        if (experiments.CollateResults.validateSingleFoldFile(fullTargetFile))
            System.out.println(exp.toShortString() + " already exists at "+fullTargetFile+", exiting.");
        else {
            Instances[] data = DatasetLoading.sampleDataset(exp.dataReadLocation, exp.datasetName, exp.foldId);
            data[0] = IntervalCreation.crop_proportional(data[0], interval[0], interval[1], normaliseInterval);
            data[1] = IntervalCreation.crop_proportional(data[1], interval[0], interval[1], normaliseInterval);

            Experiments.runExperiment(exp, data[0], data[1], classifier, fullWriteLoc);
        }
        
        return exp.classifierName;
    }
    
    
}

