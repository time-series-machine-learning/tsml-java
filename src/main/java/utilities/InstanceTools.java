/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utilities;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.ListIterator;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import java.util.TreeMap;
import utilities.class_distributions.ClassDistribution;
import utilities.class_distributions.TreeSetClassDistribution;
import utilities.generic_storage.Pair;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.DistanceFunction;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Aaron
 */
public class InstanceTools {

    public static Pair<Instance, Double> findMinDistance(Instances data, Instance inst, DistanceFunction dist){
        double min = dist.distance(data.get(0), inst);
        Instance minI = data.get(0);
        for (int i = 1; i < data.numInstances(); i++) {
            double temp = dist.distance(data.get(i), inst);
            if(temp < min){
                min = temp;
                minI = data.get(i);
            }
        }
        
        return new Pair(minI, min);
    }
    
    public static int[] deleteClassValues(Instances d){
        int[] classVals=new int[d.numInstances()];
        for(int i=0;i<d.numInstances();i++){
            classVals[i]=(int)d.instance(i).classValue();
            d.instance(i).setMissing(d.instance(i).classIndex());
        }
        return classVals;
    }
    
    /**
     * By Aaron:
     * Public method to calculate the class distributions of a dataset. Main
     * purpose is for computing shapelet qualities. 
     * 
     * @param data the input data set that the class distributions are to be
     * derived from
     * @return a TreeMap<Double, Integer> in the form of <Class Value,
     * Frequency>
     */
    public static Map<Double, Integer> createClassDistributions(Instances data)
    {
        Map<Double, Integer> classDistribution = new TreeMap<>();

        ListIterator<Instance> it = data.listIterator();
        double classValue;
        while (it.hasNext())
        {
            classValue = it.next().classValue();

            Integer val = classDistribution.get(classValue);

            val = (val != null) ? val + 1 : 1;
            classDistribution.put(classValue, val);
        }
        
        return classDistribution;
    }
    /**
     * by Tony
     * Public method to calculate the class distributions of a dataset.
     */
    public static double[] findClassDistributions(Instances data)
    {
        double[] dist=new double[data.numClasses()];
        for(Instance d:data)
            dist[(int)d.classValue()]++;
        for(int i=0;i<dist.length;i++)
            dist[i]/=data.numInstances();
        return dist;
    }
    
    /**
     * by James...
     * Public method to calculate the class distributions given a list of class labels and the number of classes.
     * Mostly to use with the data classifierresults/results analysis tools keep
     */
    public static double[] findClassDistributions(ArrayList<Double> classLabels, int numClasses)
    {
        double[] dist=new double[numClasses];
        for(double d:classLabels)
            dist[(int)d]++;
        for(int i=0;i<dist.length;i++)
            dist[i]/=classLabels.size();
        return dist;
    }
     
    public static Map<Double, Instances> createClassInstancesMap(Instances data)
    {
        Map<Double, Instances> instancesMap = new TreeMap<>();
        
        ListIterator<Instance> it = data.listIterator();
        double classValue;
        while (it.hasNext())
        {
            Instance inst = it.next();
            classValue = inst.classValue();

            Instances val = instancesMap.get(classValue);

            if(val == null)
                val = new Instances(data, 0);
            
            val.add(inst);
            
            instancesMap.put(classValue, val);
        }
        
        return instancesMap;
        
    }
    
    /** 
     * Modified from Aaron's shapelet resampling code in development.ReasamplingExperiments. Used to resample
     * train and test instances while maintaining original train/test class distributions
     * 
     * @param train Input training instances
     * @param test Input test instances
     * @param seed Used to create reproducible folds by using a consistent seed value
     * @return Instances[] with two elements; [0] is the output training instances, [1] output test instances
     */
    public static Instances[] resampleTrainAndTestInstances(Instances train, Instances test, long seed){
        if(seed==0){    //For consistency, I have made this clone the data. Its not necessary generally, but not doing it introduced a bug indiagnostics elsewhere
            Instances newTrain = new Instances(train);
            Instances newTest = new Instances(test);
            return new Instances[]{newTrain,newTest};
        }
        Instances all = new Instances(train);
        all.addAll(test);
        ClassDistribution trainDistribution = new TreeSetClassDistribution(train);
        
        Map<Double, Instances> classBins = createClassInstancesMap(all);
       
        Random r = new Random(seed);

        //empty instances.
        Instances outputTrain = new Instances(all, 0);
        Instances outputTest = new Instances(all, 0);

        Iterator<Double> keys = classBins.keySet().iterator();
        while(keys.hasNext()){
            double classVal = keys.next();
            int occurences = trainDistribution.get(classVal);
            Instances bin = classBins.get(classVal);
            bin.randomize(r); //randomise the bin.

            outputTrain.addAll(bin.subList(0,occurences));//copy the first portion of the bin into the train set
            outputTest.addAll(bin.subList(occurences, bin.size()));//copy the remaining portion of the bin into the test set.
        }

        return new Instances[]{outputTrain,outputTest};
    }

    
/**
 * 
 * @param all full data set
 * @param seed random seed so that the split can be exactly duplicated
 * @param propInTrain proportion of data for training
 * @return 
 */
    public static Instances[] resampleInstances(Instances all, long seed, double propInTrain){
        ClassDistribution classDist = new TreeSetClassDistribution(all);
        Map<Double, Instances> classBins = createClassInstancesMap(all);
       
        Random r = new Random(seed);
        //empty instances.
        Instances outputTrain = new Instances(all, 0);
        Instances outputTest = new Instances(all, 0);

        Iterator<Double> keys = classBins.keySet().iterator();
        while(keys.hasNext()){  //For each class value
            double classVal = keys.next();
            //Get the number of this class to put in train and test
            int classCount = classDist.get(classVal);
            int occurences=(int)(classCount*propInTrain);
            Instances bin = classBins.get(classVal);
            bin.randomize(r); //randomise the instances in this class.

            outputTrain.addAll(bin.subList(0,occurences));//copy the first portion of the bin into the train set
            outputTest.addAll(bin.subList(occurences, bin.size()));//copy the remaining portion of the bin into the test set.
        }

        return new Instances[]{outputTrain,outputTest};
    }
    
    
    //converts a 2d array into a weka Instances.
    public static Instances toWekaInstances(double[][] data) {
        Instances wekaInstances = null;

        if (data.length <= 0) {
            return wekaInstances;
        }

        int dimRows = data.length;
        int dimColumns = data[0].length;

        // create a list of attributes features + label
        FastVector attributes = new FastVector(dimColumns);
        for (int i = 0; i < dimColumns; i++) {
            attributes.addElement(new Attribute("attr" + String.valueOf(i + 1)));
        }

        // add the attributes 
        wekaInstances = new Instances("", attributes, dimRows);

        // add the values
        for (int i = 0; i < dimRows; i++) {
            double[] instanceValues = new double[dimColumns];

            for (int j = 0; j < dimColumns; j++) {
                instanceValues[j] = data[i][j];
            }

            wekaInstances.add(new DenseInstance(1.0, instanceValues));
        }

        return wekaInstances;
    }
    
    //converts a 2d array into a weka Instances, setting the last attribute to be the class value.
    public static Instances toWekaInstancesWithClass(double[][] data) {
        Instances wekaInstances = toWekaInstances(data);
        wekaInstances.setClassIndex(wekaInstances.numAttributes()-1);
        return wekaInstances;
    }

    
    //converts a weka Instances into a 2d array - removing class val at the end.
    public static double[][] fromWekaInstancesArray(Instances ds, boolean removeLastVal) {
        int numFeatures = ds.numAttributes() - (removeLastVal ? 1 : 0);
        int numInstances = ds.numInstances();

       double[][] data = new double[numInstances][numFeatures];

        for (int i = 0; i < numInstances; i++) {
            for (int j = 0; j < numFeatures; j++) {
                data[i][j] = ds.get(i).value(j);
            }
        }

        return data;
    }
    
        //converts a weka Instances into a 2d array.
    public static ArrayList<ArrayList<Double>> fromWekaInstancesList(Instances ds) {
        int numFeatures = ds.numAttributes()-1; //no classValue
        int numInstances = ds.numInstances();

        //Logging.println("Converting " + numInstances + " instances and " + numFeatures + " features.", LogLevel.DEBUGGING_LOG);
        ArrayList<ArrayList<Double>> data = new ArrayList<>(numInstances);
        ArrayList<Double> temp;
        for (int i = 0; i < numInstances; i++) {
            temp = new ArrayList<>(numFeatures);
            for (int j = 0; j < numFeatures; j++) {
                temp.add(ds.get(i).value(j));
            }
            data.add(temp);
        }

        return data;
    }
    
    //this is for Grabockas train/test set combo matrix. removes the need for appending.
    public static double[][] create2DMatrixFromInstances(Instances train, Instances test) {
        double [][] data = new double[train.numInstances() + test.numInstances()][train.numAttributes()];
        
        for(int i=0; i<train.numInstances(); i++)
        {
            for(int j=0; j<train.numAttributes(); j++)
            {
                data[i][j] = train.get(i).value(j);
            }
        }
        
        int index=0;
        for(int i=train.numInstances(); i<train.numInstances()+test.numInstances(); i++)
        {
            for(int j=0; j<test.numAttributes(); j++)
            {
                data[i][j] = test.get(index).value(j);
            }
            ++index;
        }
        
        return data;
    }
    
    
    // utility method for creating ARFF from UCR file without writing output, just returns Instances
    public static Instances convertFromUCRtoARFF(String inputFilePath) throws Exception{
        return convertFromUCRtoARFF(inputFilePath, null, null);
    }
    
    // writes output and returns Instances too
    public static Instances convertFromUCRtoARFF(String inputFilePath, String outputRelationName, String fullOutputPath) throws Exception{
        File input = new File(inputFilePath);
        if(!input.exists()){
            throw new Exception("Error converting to ARFF - input file not found: "+input.getAbsolutePath());
        }

        // get instance length
        Scanner scan = new Scanner(input);
        scan.useDelimiter("\n");
        String firstIns = scan.next();
        int numAtts = firstIns.split(",").length;
        
        // create attribute list
        ArrayList<Attribute> attList = new ArrayList<>();
        for(int i = 0; i < numAtts-1; i++){
            attList.add(new Attribute("att"+i));
        }
        attList.add(new Attribute("classVal"));
        
        // create Instances object
        Instances output;
        if(outputRelationName==null){
            output = new Instances("temp", attList, numAtts);
        }else{
            output = new Instances(outputRelationName, attList, numAtts);
        }
        output.setClassIndex(numAtts-1);
        
        // populate Instances
        String[] nextIns;
        DenseInstance d;
        scan = new Scanner(input);
        scan.useDelimiter("\n");
        while(scan.hasNext()){
            nextIns = scan.next().split(",");
            d = new DenseInstance(numAtts);
            for(int a = 0; a < numAtts-1; a++){
                d.setValue(a, Double.parseDouble(nextIns[a+1]));
            }
            d.setValue(numAtts-1, Double.parseDouble(nextIns[0]));
            output.add(d);
        }
        
        // if null, don't write. Else, write output ARFF here
        if(fullOutputPath!=null){
            System.out.println(fullOutputPath.substring(fullOutputPath.length()-5, fullOutputPath.length()));
            if(!fullOutputPath.substring(fullOutputPath.length()-5, fullOutputPath.length()).equalsIgnoreCase(".ARFF")){
                fullOutputPath += ".ARFF";
            }
            
            new File(fullOutputPath).getParentFile().mkdirs();
            FileWriter out = new FileWriter(fullOutputPath);
            out.append(output.toString());
            out.close();
        }
        
        return output;
    }
    public static void removeConstantTrainAttributes(Instances train, Instances test){
        int i=0;
        while(i<train.numAttributes()-1){ //Dont test class
// Test if constant
            int j=1;
            while(j<train.numInstances() && train.instance(j-1).value(i)==train.instance(j).value(i))
                j++;
            if(j==train.numInstances()){
    // Remove from train
                train.deleteAttributeAt(i);
                test.deleteAttributeAt(i);
    // Remove from test            
            }else{
                i++;
            }
        }       
    }

    
    public static boolean hasMissing(Instances ins){
        for(Instance in:ins)
            if(in.hasMissingValue())
                return true;
       return false;
    }
     //Deletes the attributes by *shifted* index
    public static void removeConstantAttributes(Instances test, int[] features){
        for(int del:features)
            test.deleteAttributeAt(del);
        
    }
    
     //Returns the *shifted* indexes, so just deleting them should work
    public static int[] removeConstantTrainAttributes(Instances train){
        int i=0;
        LinkedList<Integer> list= new LinkedList<>();
        int count=0;
        while(i<train.numAttributes()-1){ //Dont test class
// Test if constant
            int j=1;
            while(j<train.numInstances() && train.instance(j-1).value(i)==train.instance(j).value(i))
                j++;
            if(j==train.numInstances()){
    // Remove from train
                train.deleteAttributeAt(i);
                list.add(i);
    // Remove from test            
            }else{
                i++;
            }
            count++;
        }
        int[] del=new int[list.size()];
        count=0;
        for(Integer in:list){
            del[count++]=in;
        }
        return del;
        
    }
    
     //Returns the *shifted* indexes, so just deleting them should work
//Removes all constant attributes or attributes with just a single value
    public static int[] removeRedundantTrainAttributes(Instances train){
        int i=0;
        int minNumDifferent=2;
        boolean remove=false;
        LinkedList<Integer> list= new LinkedList<>();
        int count=0;
        while(i<train.numAttributes()-1){ //Dont test class
            remove=false;
// Test if constant
            int j=1;
            if(train.instance(j-1).value(i)==train.instance(j).value(i))
            while(j<train.numInstances() && train.instance(j-1).value(i)==train.instance(j).value(i))
                j++;
            if(j==train.numInstances())
                remove=true;
            else{
//Test if just a single value to remove
                count =0;
                for(j=1;j<train.numInstances();j++){
                    if(train.instance(j-1).value(i)==train.instance(j).value(i))
                        count++;
                }
                if(train.numInstances()-count<minNumDifferent+1)
                    remove=true;
            }
            if(remove)
            {
    // Remove from train
                train.deleteAttributeAt(i);
                list.add(i);
    // Remove from test            
            }else{
                i++;
            }
            count++;
        }
        int[] del=new int[list.size()];
        count=0;
        for(Integer in:list){
            del[count++]=in;
        }
        return del;
        
    }
    
    
    
    //be careful using this function. 
    //this wants to create a proportional sub sample 
    //but if you're sampling size is too small you could create a dodgy dataset.
    public static Instances subSample(Instances data, int amount, int seed){
        if(amount < data.numClasses()) System.out.println("Error: too few instances compared to classes.");

        Map<Double, Instances> classBins = createClassInstancesMap(data);
        ClassDistribution trainDistribution = new TreeSetClassDistribution(data);
        
        Random r = new Random(seed);

        //empty instances.
        Instances output = new Instances(data, 0);

        Iterator<Double> keys = classBins.keySet().iterator();
        while(keys.hasNext()){
            double classVal = keys.next();
            int occurences = trainDistribution.get(classVal);
            float proportion = (float) occurences / (float) data.numInstances();
            int numInstances = (int) (proportion * amount);
            Instances bin = classBins.get(classVal);
            bin.randomize(r); //randomise the bin.

            output.addAll(bin.subList(0,numInstances));//copy the first portion of the bin into the train set
        }

        return output;        
    }
    
    
    public static Instances subSampleFixedProportion(Instances data, double proportion, long seed){
        Map<Double, Instances> classBins = createClassInstancesMap(data);
        ClassDistribution trainDistribution = new TreeSetClassDistribution(data);
        
        Random r = new Random(seed);

        //empty instances.
        Instances output = new Instances(data, 0);

        Iterator<Double> keys = trainDistribution.keySet().iterator();
        while(keys.hasNext()){
            double classVal = keys.next();
            int occurences = trainDistribution.get(classVal);
            int numInstances = (int) (proportion * occurences);
            Instances bin = classBins.get(classVal);
            bin.randomize(r); //randomise the bin.

            output.addAll(bin.subList(0,numInstances));//copy the first portion of the bin into the train set
        }
        return output; 
     }
    
        
    //use in conjunction with subSampleFixedProportion.
    //Instances subSample = InstanceTools.subSampleFixedProportion(train, proportion, fold);
    public static double calculateSubSampleProportion(Instances train, int min){
        int small_sf = InstanceTools.findSmallestClassAmount(train);           
        double proportion = 1;
        if (small_sf>min){
            proportion = (double)min/(double)small_sf;

            if (proportion < 0.1)
                proportion = 0.1;
        }
        
        return proportion;
    }
 
    
    public static int findSmallestClassAmount(Instances data){
        ClassDistribution trainDistribution = new TreeSetClassDistribution(data);
        
        //find the smallest represented class.
        Iterator<Double> keys = trainDistribution.keySet().iterator();
        int small_sf = Integer.MAX_VALUE;
        int occurences;
        double key;
        while(keys.hasNext()){
            
            key = keys.next();
            occurences = trainDistribution.get(key);
            
            if(occurences < small_sf)
                small_sf = occurences;
        }
        
        return small_sf;
    }
    
    public static int indexOf(Instances dataset, Instance find){
        int index = -1;
        for(int i=0; i<dataset.numInstances(); i++){
            Instance in = dataset.get(i);
            boolean match = true;
            for(int j=0; j<in.numAttributes();j++){
                if(in.value(j) != find.value(j))
                    match = false;
            }
            if(match){
                index = i;
                break;
            }  
        }
        
        return index;
    }
    
    public static int indexOf2(Instances dataset, Instance find){
        int index = -1;
        for(int i=0; i< dataset.numInstances(); i++){
            if(dataset.instance(i).toString(0).contains(find.toString(0))){
                index  = i;
                break;
            }
        }
        return index;
    }
    
    
    

    
    //similar to concatinate, but interweaves the attributes. 
    //all of att_0 in each instance, then att_1 etc.
    public static Instances mergeInstances(String dataset, Instances[] inst, String[] dimChars){
        FastVector atts = new FastVector();
        String name;
        
        Instances firstInst = inst[0];
        int dimensions = inst.length;
        int length = (firstInst.numAttributes()-1)*dimensions;

        
        
        for (int i = 0; i < length; i++) {
            name = dataset + "_" + dimChars[i%dimensions] + "_" + (i/dimensions);
            atts.addElement(new Attribute(name));
        }
        
        //clone the class values over. 
        //Could be from x,y,z doesn't matter.
        Attribute target = firstInst.attribute(firstInst.classIndex());
        FastVector vals = new FastVector(target.numValues());
        for (int i = 0; i < target.numValues(); i++) {
            vals.addElement(target.value(i));
        }
        atts.addElement(new Attribute(firstInst.attribute(firstInst.classIndex()).name(), vals));
        
        //same number of xInstances 
        Instances result = new Instances(dataset + "_merged", atts, firstInst.numInstances());

        int size = result.numAttributes()-1;
        
        for(int i=0; i< firstInst.numInstances(); i++){
            result.add(new DenseInstance(size+1));
            
            for(int j=0; j<size;){
                for(int k=0; k< dimensions; k++){
                    result.instance(i).setValue(j,inst[k].get(i).value(j/dimensions)); j++;
                }
            }
        }
        
        for (int j = 0; j < result.numInstances(); j++) {
            //we always want to write the true ClassValue here. Irrelevant of binarised or not.
            result.instance(j).setValue(size, firstInst.get(j).classValue());
        }
        
        return result;
    }
}
