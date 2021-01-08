/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package utilities;

import java.io.File;
import java.io.FileWriter;
import java.util.*;

import scala.tools.nsc.Global;
import utilities.class_counts.ClassCounts;
import utilities.class_counts.TreeSetClassCounts;
import utilities.generic_storage.Pair;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Aaron
 */
public class InstanceTools {

    public static double[] countClasses(Instances data) {
        return countClasses(data, data.numClasses());
    }

    public static double[] countClasses(List<? extends Instance> data, int numClasses) {
        double[] distribution = new double[numClasses];
        for(Instance instance : data) {
            final int classValue = (int) instance.classValue();
            distribution[classValue] += instance.weight();
        }
        return distribution;
    }

    public static Map<Instance, Integer> indexInstances(Instances instances) {
        Map<Instance, Integer> instanceIntegerMap = new HashMap<>(instances.size(), 1);
        for(int i = 0; i < instances.size(); i++) {
            instanceIntegerMap.put(instances.get(i), i);
        }
        return instanceIntegerMap;
    }

    public static void setClassMissing(Instances data) {
        for(Instance instance : data) {
            instance.setClassMissing();
        }
    }

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
        if(seed==0){    //For consistency, I have made this clone the data. Its not necessary generally, but not doing it
            // introduced a bug in diagnostics elsewhere
            Instances newTrain = new Instances(train);
            Instances newTest = new Instances(test);
            return new Instances[]{newTrain,newTest};
        }
        Instances all = new Instances(train);
        all.addAll(test);
        ClassCounts trainDistribution = new TreeSetClassCounts(train);
        
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
        ClassCounts classDist = new TreeSetClassCounts(all);
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

    public static Instances resample(Instances series, double trainProportion, Random random) {
        int newSize = (int)(series.numInstances()*trainProportion);

        Instances newData = new Instances(series, newSize);
        Instances temp = new Instances(series);

        while (newData.numInstances() < newSize) {
            newData.add(temp.remove(random.nextInt(temp.numInstances())));
        }
        return newData;
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
        ArrayList<Attribute> attributes = new ArrayList<>(dimColumns);
        for (int i = 0; i < dimColumns; i++) {
            attributes.add(new Attribute("attr" + String.valueOf(i + 1)));
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


    //converts a 2d array into a weka Instances, appending the ith classlabel onto the ith row of data for each instance
    public static Instances toWekaInstances(double[][] data, double[] classLabels) {
        //todo error checking if really wanted. all utils need it at some point
        
        double[][] newData = new double[data.length][];
        
        for (int i = 0; i < data.length; i++) {
            newData[i] = new double[data[i].length + 1];
            int j = 0;
            for ( ; j < data[i].length; j++)
                newData[i][j] = data[i][j];
            
            newData[i][j] = classLabels[i];
        }
        
        return toWekaInstancesWithClass(newData);
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

/**
 * 
 * @param ins Instances object
 * @return true if there are any missing values (including class value)
 */    
    public static boolean hasMissing(Instances ins){
        for(Instance in:ins)
            if(in.hasMissingValue())
                return true;
       return false;
    }
/**
 * Deletes the attributes by *shifted* index, i.e. the positions are *not* the 
 * original positions in the data
 * @param test
 * @param features 
 */
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

    /**
     * Removes attributes deemed redundant. These are either
     * 1. All one value (i.e. constant)
     * 2. Some odd test to count the number different to the one before.
     * I think this is meant to count the number of different values?
     * It would be good to delete attributes that are identical to other attributes.
     * @param train instances from which to remove redundant attributes
     * @return array of indexes of attributes remove
     */
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
//Test pairwise similarity?
//I think this is meant to test how many different values there are. If so, it should be
//done with a HashSet of doubles. This counts how many values are identical to their predecessor
                count =0;
                for(j=1;j<train.numInstances();j++){
                    if(train.instance(j-1).value(i)==train.instance(j).value(i))
                        count++;
                }
                if(train.numInstances()-count<minNumDifferent+1)
                    remove=true;
            }
            if(remove){
    // Remove from data
                train.deleteAttributeAt(i);
                list.add(i);
            }else{
                i++;
            }
  //          count++;
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
        ClassCounts trainDistribution = new TreeSetClassCounts(data);
        
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
        ClassCounts trainDistribution = new TreeSetClassCounts(data);
        
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
        ClassCounts trainDistribution = new TreeSetClassCounts(data);
        
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
        ArrayList<Attribute> atts = new ArrayList<>();
        String name;
        
        Instances firstInst = inst[0];
        int dimensions = inst.length;
        int length = (firstInst.numAttributes()-1)*dimensions;

        
        
        for (int i = 0; i < length; i++) {
            name = dataset + "_" + dimChars[i%dimensions] + "_" + (i/dimensions);
            atts.add(new Attribute(name));
        }
        
        //clone the class values over. 
        //Could be from x,y,z doesn't matter.
        Attribute target = firstInst.attribute(firstInst.classIndex());
        ArrayList<String> vals = new ArrayList<>(target.numValues());
        for (int i = 0; i < target.numValues(); i++) {
            vals.add(target.value(i));
        }
        atts.add(new Attribute(firstInst.attribute(firstInst.classIndex()).name(), vals));
        
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

    public static void deleteClassAttribute(Instances data){
        if (data.classIndex() >= 0){
            int clsIndex = data.classIndex();
            data.setClassIndex(-1);
            data.deleteAttributeAt(clsIndex);
        }
    }

    public static List<Instances> instancesByClass(Instances instances) {
        List<Instances> instancesByClass = new ArrayList<>();
        int numClasses = instances.get(0).numClasses();
        for(int i = 0; i < numClasses; i++) {
            instancesByClass.add(new Instances(instances,0));
        }
        for(Instance instance : instances) {
            instancesByClass.get((int) instance.classValue()).add(instance);
        }
        return instancesByClass;
    }

    public static List<List<Integer>> indexByClass(Instances instances) {
        List<List<Integer>> instancesByClass = new ArrayList<>();
        int numClasses = instances.get(0).numClasses();
        for(int i = 0; i < numClasses; i++) {
            instancesByClass.add(new ArrayList());
        }
        for(int i = 0; i < instances.size(); i++) {
            instancesByClass.get((int) instances.get(i).classValue()).add(i);
        }
        return instancesByClass;
    }
    public static double[] classCounts(Instances instances) {
        double[] counts = new double[instances.numClasses()];
        for(Instance instance : instances) {
            counts[(int) instance.classValue()]++;
        }
        return counts;
    }


    public static double[] classDistribution(Instances instances) {
        double[] distribution = new double[instances.numClasses()];
        for(Instance instance : instances) {
            distribution[(int) instance.classValue()]++;
        }
        ArrayUtilities.normalise(distribution);
        return distribution;
    }
    /**
     * Concatenate features into a new Instances. Check is made that the class
     * values are the same
     * @param a
     * @param b
     * @return 
     */
    
    
    public static Instances concatenateInstances(Instances a, Instances b){
        if(a.numInstances()!=b.numInstances())
            throw new RuntimeException(" ERROR in concatenate Instances, number of cases unequal");
        for(int i=0;i<a.numInstances();i++){
            if(a.instance(i).classValue()!=b.instance(i).classValue())
                throw new RuntimeException(" ERROR in concatenate Instances, class labels not alligned in case "+i+" class in a ="+a.instance(i).classValue()+" and in b equals "+b.instance(i).classValue());
        }
            //4. Merge them all together
        Instances combo=new Instances(a);
        combo.setClassIndex(-1);
        combo.deleteAttributeAt(combo.numAttributes()-1); 
        combo=Instances.mergeInstances(combo,b);
        combo.setClassIndex(combo.numAttributes()-1);
        return combo;
    }

    public static Map<Double, Instances> byClass(Instances instances) {
        Map<Double, Instances> map = new HashMap<>();
        for(Instance instance : instances) {
            map.computeIfAbsent(instance.classValue(), k -> new Instances(instances, 0)).add(instance);
        }
        return map;
    }

    public static Instance reverseSeries(Instance inst){
        Instance newInst = new DenseInstance(inst);

        for (int i = 0; i < inst.numAttributes()-1; i++){
            newInst.setValue(i, inst.value(inst.numAttributes()-i-2));
        }

        return newInst;
    }

    public static Instance shiftSeries(Instance inst, int shift){
        Instance newInst = new DenseInstance(inst);

        if (shift < 0){
            shift = Math.abs(shift);

            for (int i = 0; i < inst.numAttributes()-shift-1; i++){
                newInst.setValue(i, inst.value(i+shift));
            }

            for (int i = inst.numAttributes()-shift-1; i < inst.numAttributes()-1; i++){
                newInst.setValue(i, 0);
            }
        }
        else if (shift > 0){
            for (int i = 0; i < shift; i++){
                newInst.setValue(i, 0);
            }

            for (int i = shift; i < inst.numAttributes()-1; i++){
                newInst.setValue(i, inst.value(i-shift));
            }
        }

        return newInst;
    }

    public static Instance randomlyAddToSeriesValues(Instance inst, Random rand, int minAdd, int maxAdd, int maxValue){
        Instance newInst = new DenseInstance(inst);

        for (int i = 0; i < inst.numAttributes()-1; i++){
            int n = rand.nextInt(maxAdd+1-minAdd)+minAdd;
            double newVal = inst.value(i)+n;
            if (newVal > maxValue) newVal = maxValue;
            newInst.setValue(i, newVal);
        }

        return newInst;
    }

    public static Instance randomlySubtractFromSeriesValues(Instance inst, Random rand, int minSubtract, int maxSubtract, int minValue){
        Instance newInst = new DenseInstance(inst);

        for (int i = 0; i < inst.numAttributes()-1; i++){
            int n = rand.nextInt(maxSubtract+1-maxSubtract)+maxSubtract;
            double newVal = inst.value(i)-n;
            if (newVal < minValue) newVal = minValue;
            newInst.setValue(i, newVal);
        }

        return newInst;
    }

    public static Instance randomlyAlterSeries(Instance inst, Random rand){
        Instance newInst = new DenseInstance(inst);

        if (rand.nextBoolean()){
            newInst = reverseSeries(newInst);
        }

        int shift = rand.nextInt(240)-120;
        newInst = shiftSeries(newInst, shift);

        if (rand.nextBoolean()){
            newInst = randomlyAddToSeriesValues(newInst, rand, 0, 5, Integer.MAX_VALUE);
        }
        else{
            newInst = randomlySubtractFromSeriesValues(newInst, rand, 0, 5, 0);
        }

        return newInst;
    }

    public static Instances shortenInstances(Instances data, double proportionRemaining, boolean normalise){
        if (proportionRemaining > 1 || proportionRemaining <= 0){
            throw new RuntimeException("Remaining series length propotion must be greater than 0 and less than or equal to 1");
        }
        else if (data.classIndex() != data.numAttributes()-1){
            throw new RuntimeException("Dataset class attribute must be the final attribute");
        }

        Instances shortenedData = new Instances(data);
        if (proportionRemaining == 1) return shortenedData;
        int newSize = (int)Math.round((data.numAttributes()-1) * proportionRemaining);
        if (newSize == 0) newSize = 1;

        if (normalise) {
            Instances tempData = truncateInstances(data, data.numAttributes()-1, newSize);
            tempData = zNormaliseWithClass(tempData);

            for (int i = 0; i < data.numInstances(); i++){
                for (int n = 0; n < tempData.numAttributes()-1; n++){
                    shortenedData.get(i).setValue(n, tempData.get(i).value(n));
                }
            }
        }

        for (int i = 0; i < data.numInstances(); i++){
            for (int n = data.numAttributes()-2; n >= newSize; n--){
                //shortenedData.get(i).setMissing(n);
                shortenedData.get(i).setValue(n, 0);
            }
        }

        return shortenedData;
    }

    public static Instances truncateInstances(Instances data, int fullLength, int newLength){
        Instances newData = new Instances(data);
        for (int i = 0; i < fullLength - newLength; i++){
            newData.deleteAttributeAt(newLength);
        }
        return newData;
    }

    public static Instances truncateInstances(Instances data, double proportionRemaining){
        if (proportionRemaining == 1) return data;
        int newSize = (int)Math.round((data.numAttributes()-1) * proportionRemaining);
        if (newSize == 0) newSize = 1;

        return truncateInstances(data, data.numAttributes()-1, newSize);
    }

    public static Instance truncateInstance(Instance inst, int fullLength, int newLength){
        Instance newInst = new DenseInstance(inst);
        for (int i = 0; i < fullLength - newLength; i++){
            newInst.deleteAttributeAt(newLength);
        }
        return newInst;
    }

    public static Instances zNormaliseWithClass(Instances data) {
        Instances newData = new Instances(data, 0);

        for (int i = 0; i < data.numInstances(); i++){
            newData.add(zNormaliseWithClass(data.get(i)));
        }

        return newData;
    }

    public static Instance zNormaliseWithClass(Instance inst){
        Instance newInst = new DenseInstance(inst);
        newInst.setDataset(inst.dataset());

        double meanSum = 0;
        int length = newInst.numAttributes()-1;

        if (length < 2) return newInst;

        for (int i = 0; i < length; i++){
            meanSum += newInst.value(i);
        }

        double mean = meanSum / length;

        double squareSum = 0;

        for (int i = 0; i < length; i++){
            double temp = newInst.value(i) - mean;
            squareSum += temp * temp;
        }

        double stdev = Math.sqrt(squareSum/(length-1));

        if (stdev == 0){
            stdev = 1;
        }

        for (int i = 0; i < length; i++){
            newInst.setValue(i, (newInst.value(i) - mean) / stdev);
        }

        return newInst;
    }

    /* Surprised these don't exist */
    public static double sum(Instance inst){
        double sumSq = 0;
        for (int j = 0; j < inst.numAttributes(); j++) {
            if (j != inst.classIndex() && !inst.attribute(j).isNominal()) {// Ignore all nominal atts
                double x = inst.value(j);
                sumSq += x;
            }
        }
        return sumSq;
    }

    public static double sumSq(Instance inst){
        double sumSq = 0;
        for (int j = 0; j < inst.numAttributes(); j++) {
            if (j != inst.classIndex() && !inst.attribute(j).isNominal()) {// Ignore all nominal atts
                double x = inst.value(j);
                sumSq += x * x;
            }
        }
        return sumSq;
    }

    public static int argmax(Instance inst){
        double max = Double.MIN_VALUE;
        int arg = -1;
        for (int j = 0; j < inst.numAttributes(); j++) {
            if (j != inst.classIndex() && !inst.attribute(j).isNominal()) {// Ignore all nominal atts{
                double x = inst.value(j);
                if (x > max){
                    max = x;
                    arg = j;
                }
            }
        }
        return arg;
    }

    public static double max(Instance inst){
        return inst.value(argmax(inst));
    }

    public static int argmin(Instance inst){
        double min = Double.MAX_VALUE;
        int arg =-1;
        for (int j = 0; j < inst.numAttributes(); j++) {
            if (j != inst.classIndex() && !inst.attribute(j).isNominal()) {// Ignore all nominal atts{
                double x = inst.value(j);
                if (x < min){
                    min = x;
                    arg = j;
                }

            }
        }
        return arg;
    }

    public static double min(Instance inst){
        return inst.value(argmin(inst));
    }



    public static double mean(Instance inst){
        double mean = 0;
        int k = 0;
        for (int j = 0; j < inst.numAttributes(); j++) {
            if (j != inst.classIndex() && !inst.attribute(j).isNominal() && !inst.isMissing(j)){// Ignore all nominal atts{
                double x = inst.value(j);
                mean +=x;
            }
            else
                k++;
        }

        return mean / (double)(inst.numAttributes() - 1 - k);
    }


    public static double variance(Instance inst, double mean){
        double var = 0;
        int k = 0;
        for (int j = 0; j < inst.numAttributes(); j++) {
            if (j != inst.classIndex() && !inst.attribute(j).isNominal()) {// Ignore all nominal atts{
                double x = inst.value(j);
                var += Math.pow(x-mean, 2);
            }
            else
                k++;
        }

        return var / (double)(inst.numAttributes() - 1 - k);
    }

    public static double kurtosis(Instance inst, double mean, double std){
        double kurt = 0;
        int k = 0;
        for (int j = 0; j < inst.numAttributes(); j++) {
            if (j != inst.classIndex() && !inst.attribute(j).isNominal()) {// Ignore all nominal atts{
                double x = inst.value(j);
                kurt += Math.pow(x-mean, 4);
            }
            else
                k++;
        }

        kurt /= Math.pow(std, 4);
        return kurt / (double)(inst.numAttributes() - 1 - k);
    }


    public static double skew(Instance inst, double mean, double std){
        double skew = 0;
        int k = 0;
        for (int j = 0; j < inst.numAttributes(); j++) {
            if (j != inst.classIndex() && !inst.attribute(j).isNominal()) {// Ignore all nominal atts{
                double x = inst.value(j);
                skew += Math.pow(x-mean, 3);
            }
            else
                k++;
        }

        skew /= Math.pow(std, 3);
        return skew / (double)(inst.numAttributes() - 1 - k);
    }

    public static double slope(Instance inst, double sum, double sumXX){
        double sumXY= 0;
        int k =0;
        for (int j = 0; j < inst.numAttributes(); j++) {
            if (j != inst.classIndex() && !inst.attribute(j).isNominal()) {// Ignore all nominal atts{
                sumXY += inst.value(j) *  j;
            }
            else
                k++;
        }
        double length = (double)(inst.numAttributes() - k);

        double sqsum = sum*sum;
        //slope
        double slope = sumXY-sqsum/length;
        double denom=sumXX-sqsum/length;
        if(denom!=0)
            slope/=denom;
        else
            slope=0;

        //if(standardDeviation == 0)
        //slope=0;

        return slope;
    }

    public static double[] ConvertInstanceToArrayRemovingClassValue(Instance inst, int c) {
        double[]  d = inst.toDoubleArray();
        double[] temp;

		if (c >= 0) {
			temp = new double[d.length - 1];
			System.arraycopy(d, 0, temp, 0, c);
			d = temp;
        }

        return d;

    }

    public static double[] ConvertInstanceToArrayRemovingClassValue(Instance inst) {
        return ConvertInstanceToArrayRemovingClassValue(inst, inst.classIndex());
    }

}
