/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utilities.multivariate_tools;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Scanner;
import java.util.Set;
import static utilities.GenericTools.indexOf;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instances;

import static utilities.ClassifierTools.loadData;
import weka.core.Instance;
/**
 *
 * @author raj09hxu
 */
public class ConvertDatasets {
    
    static int classIndex = 0;
    public static void main(String[] args) throws FileNotFoundException {
        
        //createAALTD();
        //createCricket();
        //createArticularyWord();
        //createVillar();
        createMVMotion();
        
        /*createSingleDatasets("E:\\LocalData\\Dropbox\\Multivariate TSC\\Aarons Official\\","UWaveGesture", 0);
        createSingleDatasets("E:\\LocalData\\Dropbox\\Multivariate TSC\\Aarons Official\\","HandwritingGyroscope", 0);
        createSingleDatasets("E:\\LocalData\\Dropbox\\Multivariate TSC\\Aarons Official\\","HandwritingAccelerometer", 0);
        createSingleDatasets("E:\\LocalData\\Dropbox\\Multivariate TSC\\Aarons Official\\","Cricket",0);
        createSingleDatasets("E:\\LocalData\\Dropbox\\Multivariate TSC\\Aarons Official\\", "ArticularyWord",0);
        */
        
        //createSingleDatasets("","DTW_A_Test",0);
        
        /*String dir = "E:\\LocalData\\Dropbox\\Multivariate TSC\\Aarons Official\\ArticularyWord\\";
        
        Instances[] LL = splitMultivariateInstances(loadData(dir+"ArticularyWordLL")); 
        Instances[] T1 = splitMultivariateInstances(loadData(dir+"ArticularyWordT1"));
        Instances[] UL = splitMultivariateInstances(loadData(dir+"ArticularyWordUL"));
        
        List<Instances> list = new ArrayList(LL.length+T1.length+UL.length);
        list.addAll(Arrays.asList(LL));
        list.addAll(Arrays.asList(T1));
        list.addAll(Arrays.asList(UL));
        saveDataset(mergeToMultivariateInstances(list.toArray(new Instances[list.size()])), dir+"ArticularyWord");
        
        Instances data = utilities.ClassifierTools.loadData(dir  + "ArticularyWord.arff");

        Instances train, test;

        Instances[] train_test = utilities.MultivariateInstanceTools.resampleMultivariateInstances(data, 0, 0.5);
        train = train_test[0];
        test = train_test[1];


        utilities.ClassifierTools.saveDataset(train, dir + "ArticularyWord" + "_TRAIN");
        utilities.ClassifierTools.saveDataset(test, dir + "ArticularyWord" + "_TEST");*/
        
        /*saveDataset(mergeToMultivariateInstances(new Instances[]{
            loadData("DTW_A_TEST/A_TRAIN"), 
            loadData("DTW_A_TEST/B_TRAIN")}
        ), "AB_TRAIN");
        
        saveDataset(mergeToMultivariateInstances(new Instances[]{
            loadData("DTW_A_TEST/A_TEST"), 
            loadData("DTW_A_TEST/B_TEST")}
        ), "AB_TEST");*/
        
        
        /*String[] end = {"X", "Y", "Z"};
        String[] trainTest = {"LL", "T1", "UL"};
        String dir = "E:\\LocalData\\Dropbox\\Multivariate TSC\\Aarons Official\\";
        String dataset = "ArticularyWord";
        
        for(String type : trainTest){
            
            Instances[] data = new Instances[end.length];
            int i=0;
            for(String en : end){
                data[i++] = utilities.ClassifierTools.loadData(dir + dataset + "\\" + dataset + type + en);
            }
            
            utilities.ClassifierTools.saveDataset(utilities.MultivariateInstanceTools.mergeToMultivariateInstances(data), dir + dataset + "\\" + dataset + type + Arrays.toString(end).replace("[", "").replace("]", "").replace(",", "").replace("\\s+", ""));
        }*/
    }
    
    static void createAALTD(){
                
        String[] end = {"X", "Y", "Z"};
        String dir = "E:\\LocalData\\Dropbox\\Multivariate TSC\\Aarons Official\\";
        String dataset = "AALTD";
        
        for(int i=0; i<8; i++){
            
            Instances[] data = new Instances[end.length];
            int j=0;
            for(String en : end){
                data[j++] = utilities.ClassifierTools.loadData(dir + dataset + "\\univariate\\" + dataset + "_"+i+"_" + en + "_TRAIN");
            }
            
            Instances merged = utilities.multivariate_tools.MultivariateInstanceTools.mergeToMultivariateInstances(data);
            utilities.ClassifierTools.saveDataset(merged, dir + dataset + "\\" + dataset + "_" +i);
        
            //split into train and test
            Instances[] train_test = utilities.multivariate_tools.MultivariateInstanceTools.resampleMultivariateInstances(merged, 0, 0.5);
            utilities.ClassifierTools.saveDataset(train_test[0], dir + dataset + "\\" + dataset + "_" + i + "_TRAIN");
            utilities.ClassifierTools.saveDataset(train_test[1], dir + dataset + "\\" + dataset + "_" + i + "_TEST");
        }
    }
    
    static void createCricket(){            
        String[] end = {"X", "Y", "Z"};
        String dir = "E:\\LocalData\\Dropbox\\Multivariate TSC\\Aarons Official\\";
        String dataset = "Cricket";

        for(String LR : new String[]{"Left", "Right"}){
            
            Instances[] data = new Instances[end.length];
            int j=0;
            for(String end1 : end){
                data[j++] = utilities.ClassifierTools.loadData(dir + dataset + "\\univariate\\" + dataset +end1+ LR);
            }
            
            Instances merged = utilities.multivariate_tools.MultivariateInstanceTools.mergeToMultivariateInstances(data);
            utilities.ClassifierTools.saveDataset(merged, dir + dataset + "\\" + dataset + "_" + LR);
        
            //split into train and test
            Instances[] train_test = utilities.multivariate_tools.MultivariateInstanceTools.resampleMultivariateInstances(merged, 0, 0.5);
            utilities.ClassifierTools.saveDataset(train_test[0], dir + dataset + "\\" + dataset + "_" + LR + "_TRAIN");
            utilities.ClassifierTools.saveDataset(train_test[1], dir + dataset + "\\" + dataset + "_" + LR + "_TEST");
        }
    }
    
    public static void createArticularyWord(){
                String[] end = {"X", "Y", "Z"};
        String dir = "E:\\LocalData\\Dropbox\\Multivariate TSC\\Aarons Official\\";
        String dataset = "ArticularyWord";

        for(String LR : new String[]{"LL", "T1", "UL"}){
            
            Instances[] data = new Instances[end.length];
            int j=0;
            for(String end1 : end){
                data[j++] = utilities.ClassifierTools.loadData(dir + dataset + "\\univariate\\" + dataset + LR+end1);
            }
            
            Instances merged = utilities.multivariate_tools.MultivariateInstanceTools.mergeToMultivariateInstances(data);
            utilities.ClassifierTools.saveDataset(merged, dir + dataset + "\\" + dataset + "_" + LR);
        
            //split into train and test
            Instances[] train_test = utilities.multivariate_tools.MultivariateInstanceTools.resampleMultivariateInstances(merged, 0, 0.5);
            utilities.ClassifierTools.saveDataset(train_test[0], dir + dataset + "\\" + dataset + "_" + LR + "_TRAIN");
            utilities.ClassifierTools.saveDataset(train_test[1], dir + dataset + "\\" + dataset + "_" + LR + "_TEST");
        }
        
        
    }
    
    //not yet ready.
    public static void createVillar(){
        String dir = "E:\\LocalData\\Dropbox\\Multivariate TSC\\Aarons Official\\";
        String dataset = "VillarData";
        Instances[] train_test = utilities.multivariate_tools.MultivariateInstanceTools.resampleMultivariateInstances(loadData(dir + dataset + "\\" + dataset), 0, 0.5);
        utilities.ClassifierTools.saveDataset(train_test[0], dir + dataset + "\\" + dataset + "_TRAIN");
        utilities.ClassifierTools.saveDataset(train_test[1], dir + dataset + "\\" + dataset + "_TEST");
    }
    
    public static void createMVMotion(){
        String dir = "E:\\LocalData\\Dropbox\\Multivariate TSC\\Aarons Official\\";
        
        //load up the MVMotion2 dataset. Split it into accelormeter and gyro data, then combine and split into 50/50 splits.
        String dir2 = "E:\\LocalData\\Dropbox\\Multivariate TSC\\Aarons Official\\Old data\\MVMotion\\MVMotion2.arff";
        Instances data = utilities.ClassifierTools.loadData(dir2);
        Instances[] data_channels = utilities.multivariate_tools.MultivariateInstanceTools.splitMultivariateInstances(data);
        
        
        //copy from's to is exclusive
        //0,1,2 should be accel, 3,4,5 should gyro.
        Instances[] accel_channels = Arrays.copyOfRange(data_channels, 0, 3);
        Instances accel = utilities.multivariate_tools.MultivariateInstanceTools.mergeToMultivariateInstances(accel_channels);
        Instances[] gyro_channels = Arrays.copyOfRange(data_channels, 3, 6);
        Instances gyro = utilities.multivariate_tools.MultivariateInstanceTools.mergeToMultivariateInstances(gyro_channels);
        
        //create MVMotionA train test
        String dataset = "MVMotionA";
        Instances[] train_test = utilities.multivariate_tools.MultivariateInstanceTools.resampleMultivariateInstances(accel, 0, 0.5);
        utilities.ClassifierTools.saveDataset(train_test[0], dir + dataset + "\\" + dataset + "_TRAIN");
        utilities.ClassifierTools.saveDataset(train_test[1], dir + dataset + "\\" + dataset + "_TEST");
        
        //create MVMotionAG train test
        dataset = "MVMotionAG";
        train_test = utilities.multivariate_tools.MultivariateInstanceTools.resampleMultivariateInstances(data, 0, 0.5);
        utilities.ClassifierTools.saveDataset(train_test[0], dir + dataset + "\\" + dataset + "_TRAIN");
        utilities.ClassifierTools.saveDataset(train_test[1], dir + dataset + "\\" + dataset + "_TEST");

        //create MVMotionG train test and extract the G part from MVMotionG
        //create MVMotionA train test
        dataset = "MVMotionG";
        train_test = utilities.multivariate_tools.MultivariateInstanceTools.resampleMultivariateInstances(gyro, 0, 0.5);
        utilities.ClassifierTools.saveDataset(train_test[0], dir + dataset + "\\" + dataset + "_TRAIN");
        utilities.ClassifierTools.saveDataset(train_test[1], dir + dataset + "\\" + dataset + "_TEST");
    }

    
    public static void createSingleDatasets(String dir1, String dirName, int classInd) throws FileNotFoundException{
        classIndex = classInd;
        File dir = new File(dir1+dirName+"\\");
        for(File f : dir.listFiles()){
            if(f.isDirectory()) continue;
            Instances data = createArff(f);
            utilities.ClassifierTools.saveDataset(data, dir + "\\"+dirName+"_"+f.getName());
        }
    }
    
    public static Instances createArff(File f_in) throws FileNotFoundException{
        Scanner sc = new Scanner(f_in);
        
        List<double[]> timeseries = new ArrayList();
        List<Double> classVals = new ArrayList();
        
        //classIndex is in position 0.
        while(sc.hasNextLine()){
            String line = sc.nextLine();
            String[] data = line.split("\\s+");
            
            //if the class value is not at the beginning, it's at the end.
            int start = classIndex == 0 ? 1 : 0;
            int end = classIndex == 0 ? data.length : data.length-1;
            
            double[] d = new double[data.length-1];
            int i=0;
            for(; start < end; start++){
                d[i++] = Double.parseDouble(data[start]);
            }
            
            timeseries.add(d);
            classVals.add(Double.parseDouble(data[classIndex]));
        }
        
        
        return buildArff(timeseries, classVals.stream().mapToDouble(i->i).toArray());
    }
    
    public static Instances buildArff(List<double[]> dataRows, double[] classVals){
        
        //build the attributes.
        Instances output = null;

        int dimCols = dataRows.get(0).length;
        int dimRows = dataRows.size();
        
        // create a list of attributes features + label
        FastVector attributes = new FastVector();
        for (int i = 0; i < dimCols; i++) {
            attributes.addElement(new Attribute("attr" + String.valueOf(i + 1)));
        }
        
        //also add the classValue.
        //figure out how many classValues there are.
        double[] values = uniqueValues(classVals);
        Arrays.sort(values);
        FastVector vals = new FastVector(values.length);
        for (int i = 0; i < values.length; i++) {
            vals.addElement(""+values[i]);
        }
        attributes.addElement(new Attribute("classAttribute", vals));
        
        // add the attributes 
        output = new Instances("", attributes, dataRows.size());

        // add the values
        for (int i = 0; i < dimRows; i++) {
            output.add(new DenseInstance(dimCols + 1));
            for(int j=0; j<dimCols; j++){
                output.instance(i).setValue(j, dataRows.get(i)[j]);
            }
            
            //set class value at the end.
            output.instance(i).setValue(dimCols, indexOf(values,classVals[i]));
        }
        
        output.setClassIndex(output.numAttributes()-1);
        
        return output;
    }
    
    public static double[] uniqueValues(double[] classVals){
        Set<Double> set = new HashSet();
        for(double d : classVals)
            set.add(d);
        return set.stream().mapToDouble(Double::doubleValue).toArray();
    }
       
    
}
