/*
 * Created on Dec 4, 2005
 
 */
package utilities;


import java.io.FileReader;
import java.util.ArrayList;
import java.util.Random;
import weka.classifiers.*;
import weka.classifiers.bayes.*;

import weka.classifiers.evaluation.EvaluationUtils;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;


import fileIO.OutFile;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import statistics.distributions.NormalDistribution;
import weka.classifiers.lazy.kNN;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.converters.ArffSaver;

/**
 * @author ajb
 *
 *Methods to perform Classification tasks with Weka which I cant seem to
 *do in Weka 
 */
public class ClassifierTools {
	
        /** 
         * simply loads the file on path or exits the program
         * @param fullPath source path for ARFF file WITHOUT THE EXTENSION for some reason
         * @return Instances from path
         */
	public static Instances loadData(String fullPath){
            if(!fullPath.toLowerCase().endsWith(".arff"))
                fullPath += ".arff";
        
            try {
                return loadData(new File(fullPath));
            } catch(IOException e) {
                System.out.println("Unable to load data on path "+fullPath+" Exception thrown ="+e);
                return null;
            }
	}
	
        public static Instances loadDataThrowable(String fullPath) throws IOException{
            if(!fullPath.toLowerCase().endsWith(".arff"))
                fullPath += ".arff";
        
            return loadData(new File(fullPath));
	}
        
        /** 
        * simply loads the instances from the file
        * @param file the File pointer rather than the path. Useful if you use FilenameFilters.
        * @return Instances from file.
        */
        public static Instances loadData(File file) throws IOException{
            FileReader reader = new FileReader(file);
            Instances inst = new Instances(reader);
            inst.setClassIndex(inst.numAttributes()-1);
            reader.close();
            return inst;
        }
        
    /**
     *  Simple util to saveDatasets out. Useful for shapelet transform.
     * 
     * @param dataSet
     * @param fileName
     */
    public static void saveDataset(Instances dataSet, String fileName)
    {
        try
        {
            ArffSaver saver = new ArffSaver();
            saver.setInstances(dataSet);
            if (fileName.endsWith(".arff"))
                saver.setFile(new File(fileName));
            else 
                saver.setFile(new File(fileName + ".arff"));
            saver.writeBatch();
        }
        catch (IOException ex)
        {
            System.out.println("Error saving transformed dataset" + ex);
        }
    }

/**
 * 	Simple util to find the accuracy of a trained classifier on a test set. Probably is a built in method for this! 
 * @param test
 * @param c
 * @return accuracy of classifier c on Instances test
 */
	public static double accuracy(Instances test, Classifier c){
		double a=0;
		int size=test.numInstances();
		Instance d;
		double predictedClass,trueClass;
		for(int i=0;i<size;i++)
		{
			d=test.instance(i);
			try{
				predictedClass=c.classifyInstance(d);
				trueClass=d.classValue();
				if(trueClass==predictedClass)
					a++;
//				System.out.println("True = "+trueClass+" Predicted = "+predictedClass);
			}catch(Exception e){
                            System.out.println(" Error with instance "+i+" with Classifier "+c.getClass().getName()+" Exception ="+e);
                            e.printStackTrace();
                            System.exit(0);
                        }
		}
		return a/size;
	}	
	
	public static Classifier[] setDefaultSingleClassifiers(ArrayList<String> names){
		ArrayList<Classifier> sc2=new ArrayList<>();
		sc2.add(new kNN(1));
		names.add("NN");
		Classifier c;
		sc2.add(new NaiveBayes());
		names.add("NB");
		sc2.add(new J48());
		names.add("C45");
		c=new SMO();
		PolyKernel kernel = new PolyKernel();
		kernel.setExponent(1);
		((SMO)c).setKernel(kernel);
		sc2.add(c);
		names.add("SVML");
		c=new SMO();
		kernel = new PolyKernel();
		kernel.setExponent(2);
		((SMO)c).setKernel(kernel);
		sc2.add(c);
		names.add("SVMQ");
		c=new RandomForest();
		((RandomForest)c).setNumTrees(100);
		sc2.add(c);
		names.add("RandF100");
		c=new RotationForest();
		sc2.add(c);
		names.add("RotF30");
	
		Classifier[] sc=new Classifier[sc2.size()];
		for(int i=0;i<sc.length;i++)
			sc[i]=sc2.get(i);

		return sc;
	}
         
        
/**
 * This method returns the data in the same order it was given, and returns probability distributions for each test data
 * Assume data is randomised already
 * @param trainData
 * @param testData
 * @param c
 * @return distributionForInstance for each Instance in testData
 */
	public static double[][] predict(Instances trainData,Instances testData, Classifier c){
		double[][] results=new double[testData.numInstances()][];
		try{
			c.buildClassifier(trainData);
			for(int i=0;i<testData.numInstances();i++)
				results[i]=c.distributionForInstance(testData.instance(i));
		}catch(Exception e){
			System.out.println(" Error in manual cross val");
		}
		return results;
	}
/**
 * This method does a cross validation using the EvaluationUtils and stores the predicted and actual values.
 * I implemented this because I saw no way of using the built in cross vals to get the actual predictions,
 * useful for e.g McNemar's test (and cv variance). Note that the use of FastVector has been depreciated 
 * @param c
 * @param allData
 * @param m
 * @return
 */
    @SuppressWarnings({ "deprecation", "rawtypes" })
    public static double[][] crossValidation(Classifier c, Instances allData, int m){
            EvaluationUtils evalU;
            double[][] preds=new double[2][allData.numInstances()];
            Object[] p;
            FastVector f;
            NominalPrediction nom;
            try{
                    evalU=new EvaluationUtils();
                    evalU.setSeed(10);
                    f=evalU.getCVPredictions(c,allData,m);
                    p=f.toArray(); 
                    for(int i=0;i<p.length;i++)
                    {
                            nom=((NominalPrediction)p[i]);
                            preds[1][i]=nom.predicted();
                            preds[0][i]=nom.actual();
                    }
            }catch(Exception e){
                    System.out.println(" Error ="+e+" in method Cross Validate Experiment");
                    e.printStackTrace();
                    System.out.println(allData.relationName());
                    System.exit(0);

            }
            return preds;

    }
	
	
/**
* 	This method does a cross validation using the EvaluationUtils and stores t
* he predicted and actual values.
* Accuracy is stored in preds[0][0], StdDev of accuracy between folds SHOULD BE 
* stored in preds[1][0].
* TO IMPLEMENT!
* Could do with some testing, there is some uncertainty over the last fold.
* @param allData
* @param m
* @return
*/
    @SuppressWarnings({ "deprecation", "rawtypes" })
    public static double[][] crossValidationWithStats(Classifier c, Instances allData, int m)
    {
            EvaluationUtils evalU;
            double[][] preds=new double[2][allData.numInstances()+1];
            int foldSize=allData.numInstances()/m;  //Last fold may have fewer cases than this
            FastVector f;
            Object[] p;
            NominalPrediction nom;
            double acc=0,sum=0,sumsq=0;
            try{
                    evalU=new EvaluationUtils();
//				evalU.setSeed(10);
                    f=evalU.getCVPredictions(c,allData,m);
                    p=f.toArray(); 
                    for(int i=0;i<p.length;i++)
                    {
                        nom=((NominalPrediction)p[i]);
                        preds[1][i+1]=nom.predicted();
                        preds[0][i+1]=nom.actual();
//					System.out.println(" pred = "+preds[i+1]);
                        if(preds[0][i+1]==preds[1][i+1]){
                            preds[0][0]++;
                            acc++;
                        }
                        if((i>0 && i%foldSize==0)){
//Sum Squares                                        
                            sumsq+=(acc/foldSize)*(acc/foldSize);
//Sum                                                                           
                            sum+=(acc/foldSize);
                            acc=0;                                        
                        }
                    }
                    //Accuracy stored in preds[0][0]
                    preds[0][0]=preds[0][0]/p.length;
                    preds[1][0]=(sumsq-sum*sum/m)/m;
                    preds[1][0]=Math.sqrt(preds[1][0]);
            }catch(Exception e)
            {
                    System.out.println(" Error ="+e+" in method Cross Validate Experiment");
                    e.printStackTrace();
                    System.out.println(allData.relationName());
                    System.exit(0);

            }
            return preds;

    }
		
    public static double stratifiedCrossValidation(Instances data, Classifier c, int folds, int seed){
        Random rand = new Random(seed);   // create seeded number generator
        Instances randData = new Instances(data);   // create copy of original data
        randData.randomize(rand);         // randomize data with number generator
        randData.stratify(folds);
        int correct=0;
        int total=data.numInstances();
        for (int n = 0; n < folds; n++) {
           Instances train = randData.trainCV(folds, n);
           Instances test = randData.testCV(folds, n);
           try{
               c.buildClassifier(train);
                for(Instance ins:test){
                    int pred=(int)c.classifyInstance(ins);
                    if(pred==ins.classValue())
                        correct++;
                }
//                System.out.println("Finished fold "+n+" acc ="+((double)correct/((n+1)*test.numInstances())));
           }catch(Exception e){
               System.err.println("ERROR BUILDING FOLD "+n+" for data set "+data.relationName());
               e.printStackTrace();
               System.exit(1);
           }
        }            
        return ((double)correct)/total;
    }

/**
 * This does a manual cross validation (i.e. without EvalUtils) and rather confusingly returns 
 * the distribution for each Instance (as opposed to the predicted/actual in method performCrossValidation
 * @param data
 * @param c
 * @param numFolds
 * @return distribution for each Instance 
 */
    public static double[][] performManualCrossValidation(Instances data, Classifier c, int numFolds)
    {
        double[][] results=new double[data.numInstances()][data.numClasses()];
        Instances train;
        Instances test;
        int interval = data.numInstances()/numFolds;
        int start=0;		
        int end=interval;
        int testCount=0;
        try{
            for(int f=0;f<numFolds;f++){
                //Split Data
                train=new Instances(data,0);
                test=new Instances(data,0);
                for(int i=0;i<data.numInstances();i++){
                    if(i>=start && i<end)
                            test.add(data.instance(i));
                    else
                            train.add(data.instance(i));
                }
                //Classify on training
                c.buildClassifier(data);
                //Predict
                for(int i=0;i<interval;i++){
                    results[testCount]=c.distributionForInstance(test.instance(i));
                    testCount++;
                }
                //Increment
                start=end;
                end=end+interval;
            }
        }catch(Exception e){
            System.out.println(" Error in manual cross val");
        }
        return results;
    }
	
/**
 * Writes the predictions vs actual of a pre trained classifier to a file
 * @param model
 * @param data
 * @param path
 */
    public static void makePredictions(Classifier model,Instances data, String path){
        OutFile f1 = new OutFile(path+".csv");
        double actual,pred;
        Instance t;
        try{
            for(int i=0;i<data.numInstances();i++){
                t=data.instance(i);
                actual=t.classValue();
                pred=model.classifyInstance(t);
                f1.writeLine(i+","+actual+","+pred);
            }
        }catch(Exception e){
            System.out.println("Exception in makePredictions"+e);
        }
    }

    public static Classifier[] setSingleClassifiers(ArrayList<String> names){
        ArrayList<Classifier> sc2=new ArrayList<Classifier>();
        IBk k=new IBk(50);
        k.setCrossValidate(true);
        sc2.add(k);
        names.add("kNN");
        Classifier c;
        sc2.add(new NaiveBayes());
        names.add("NB");
        sc2.add(new J48());
        names.add("C45");
        c=new SMO();
        PolyKernel kernel = new PolyKernel();
        kernel.setExponent(1);
        ((SMO)c).setKernel(kernel);
        sc2.add(c);
        names.add("SVML");
        c=new SMO();
        kernel = new PolyKernel();
        kernel.setExponent(2);
        ((SMO)c).setKernel(kernel);
        sc2.add(c);
        names.add("SVMQ");
        c=new RandomForest();
        ((RandomForest)c).setNumTrees(100);
        sc2.add(c);
        names.add("RandF100");
        c=new RotationForest();
        sc2.add(c);
        names.add("RotF30");

        Classifier[] sc=new Classifier[sc2.size()];
        for(int i=0;i<sc.length;i++)
                sc[i]=sc2.get(i);

        return sc;
    }

    public static double singleTrainTestSplitAccuracy(Classifier c, Instances train, Instances test){
        //Perform a simple experiment,
        double acc=0;
        try{
            c.buildClassifier(train);
            int correct=0;
            for(Instance ins:test){
                int pred=(int)c.classifyInstance(ins);
//                System.out.println((int)ins.classValue()+","+pred);
                if(pred==(int)ins.classValue())
                    correct++;
            }
            acc=correct/(double)test.numInstances();
        }catch(Exception e)
        {
            System.out.println(" Error ="+e+" in method singleTrainTestSplitAccuracy"+e);
            e.printStackTrace();
            System.exit(0);
        }
        return acc;
    }	
    
 /* Returns probability distribution for each instance with no randomisation  
    */
    public static double[][] crossValidate(Classifier c,Instances data,  int numFolds)
    {
        double[][] results=new double[data.numInstances()][data.numClasses()];
        Instances train;
        Instances test;
        int interval = data.numInstances()/numFolds;
        int start=0;		
        int end=interval;
        int testCount=0;
        try{
            for(int f=0;f<numFolds;f++){
                if(f==numFolds-1)
                    end=data.numInstances();
                //Split Data
                train=new Instances(data,0);
                test=new Instances(data,0);
                for(int i=0;i<data.numInstances();i++){
                    if(i>=start && i<end)
                            test.add(data.instance(i));
                    else
                            train.add(data.instance(i));
                }
                //Classify on training
                c.buildClassifier(train);
                //Predict
                for(int i=0;i<test.numInstances();i++){
                    results[testCount]=c.distributionForInstance(test.instance(i));
                    testCount++;
                }
                //Increment
                start=end;
                end=end+interval;
            }
        }catch(Exception e){
            System.out.println(" Error in manual cross val");
        }
        return results;
    }
    
    
    public static ClassifierResults constructClassifierResults(Classifier classifier, Instances test) throws Exception{
        double[] preds = new double[test.numInstances()];
        double[][] distForInstances = new double[test.numInstances()][];
        double correct = 0;
        for(int i=0; i<test.numInstances(); i++){
            Instance test1 = test.get(i);
            distForInstances[i] = classifier.distributionForInstance(test1);
            preds[i] = utilities.GenericTools.indexOfMax(distForInstances[i]);
            if(preds[i] == test1.classValue())
                correct++;
        }
        
        double accuracy = correct / (double) test.numInstances();
        double[] classVals = test.attributeToDoubleArray(test.classIndex());
        ClassifierResults results = new ClassifierResults(accuracy, classVals, preds, distForInstances, test.numClasses());
        results.setNumInstances(test.numInstances());
        results.setNumClasses(test.numClasses());
        return results;
    }
	
    
    public static class ResultsStats{
        public double accuracy;
        public double sd;
        public double min;
        public double max;

        public ResultsStats(){
            accuracy=0;
            sd=0;
        }
        public ResultsStats(double[][] preds, int folds){
            findCVMeanSD(preds,folds);
        }
        public static ResultsStats find(double[][] preds, int folds){
            ResultsStats f=new ResultsStats();
            f.findCVMeanSD(preds,folds);
            return f;
        }
        public void findCVMeanSD(double[][] preds, int folds){
            double[] acc= new double[folds];
            //System.out.println("No. of folds = "+acc.length); // Test output
            int count=0; // Changed from 1
            int window=(preds[0].length-1)/folds;	//Put any excess in the last fold 
            window=(preds[0].length)/folds; //Changed from length-1
            //System.out.println("Window = "+window+" readings; excess goes in last fold"); // Test output
            for(int i=0;i<folds-1;i++){
                acc[i]=0;
                for(int j=0;j<window;j++){
                        if(preds[0][count]==preds[1][count])
                                acc[i]++;
                        count++;
                }
            }
            //Last fold is the remainder 
            int lastSize=preds[0].length-count;
            //System.out.println("Last fold has " + lastSize + " instances.");//Test output
            for(int j=count;j<preds[0].length;j++){
                if(preds[0][count]==preds[1][count])
                        acc[folds-1]++;
                count++;
            }
            //System.out.println("Final fold has accuracy = " + acc[folds-1]);//Test outputs
//Find mean, min and max		
            accuracy=acc[0];
            //System.out.println("First fold accuracy = "+accuracy);//Test output
            //min=1.0; // Should be acc[0];
            min=acc[0];
            max=0;
            for(int i=1;i<folds;i++){
                accuracy+=acc[i];
                //System.out.println("Sum of accuracies = " + accuracy);//Test output
                if(acc[i]<min)
                        min=acc[i];
                if(acc[i]>max)
                        max=acc[i];
            }
            //System.out.println(accuracy+"/"+(preds[0].length));//Test output
            accuracy/=preds[0].length;//Changed from length -1.
            //System.out.println(accuracy);//Test output
//Find SD
            sd=0;
            for(int i=0;i<folds-1;i++)// Changed from int i=1
            {
                sd+=(acc[i]/window-accuracy)*(acc[i]/window-accuracy);
             //System.out.println("Accuracy used here = " +acc[i]); //Test output, added braces.
            }
            sd+=(acc[folds-1]/lastSize-accuracy)*(acc[folds-1]/lastSize-accuracy);//Last fold
            sd/=folds;
            sd=Math.sqrt(sd);
        }
        public String toString(){
                return "Accuracy = "+accuracy+" SD = "+sd+" Min = "+min+" Max = "+max; //Added some spaces
        }
    }
	
//	public static void main(String[] args) // Test harness.
//	{
//		
//            double[][] preds = {
//                                {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0},
//                                {1.0, 5.0, 3.0,4.0,5.0,6.0,3.0,8.0,9.0}
//                               };
//            
//            double[][] preds2 = {
//                                {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0},
//                                {1.4, 5.0, 3.0,4.0,5.0,6.0,3.0,8.0,9.0}
//                               };
//            int folds = 3;
//            
//            //folds = preds[0].length;
//            
//            //System.out.println(folds);
////            
////            
//            
//            System.out.println("preds");
//            ResultsStats rs = new ResultsStats(preds,folds);
//            
//            System.out.println(rs);
//            
//            System.out.println("preds2");
//            rs = new ResultsStats(preds2,folds);
//            
//            System.out.println(rs);
//		
//	}
		

/********** Some other random methods *****************/
	//If the folds is 1, do a simple test/train split. Otherwise, do a cross validation by first combining the sets	
        public static ResultsStats[] evalClassifiers(Instances test, Instances train, int folds,Classifier[] sc) throws Exception{
                int nosClassifiers=sc.length;
                double[][]  preds;
                ResultsStats[] mean=new ResultsStats[nosClassifiers];
                int seed=100;

                for(int i=0;i<nosClassifiers;i++){
//				String[] settings=sc[i].getOptions();
//				for(String s:settings)
//					System.out.print(","+s);

//				System.out.print("\t folds ="+folds);
                        if(folds>1){	// Combine the two files
                                Instances full=new Instances(train);//Instances.mergeInstances(train, test);
                                for(int j=0;j<test.numInstances();j++)
                                        full.add(test.instance(j));
                    Random rand = new Random(seed);
//					System.out.print("\t cases ="+full.numInstances());
                    full.randomize(rand);
                                preds=crossValidation(sc[i],full,folds);
                                mean[i]= ResultsStats.find(preds,full.numInstances());
//					System.out.println("\t : "+mean[i].accuracy);
                        }
                        else{
                                sc[i].buildClassifier(train);
                                mean[i]=new ResultsStats();
                                mean[i].accuracy=accuracy(test,sc[i]);
//					System.out.println("\t : "+mean[i].accuracy);
                        }
                }
 		return mean;
	}		

       
        
	public static Instances estimateMissing(Instances data){

		ReplaceMissingValues nb = new ReplaceMissingValues();
		Instances nd=null;
		try{
			nb.setInputFormat(data);
			Instance temp;
			int n = data.numInstances();
			for(int i=0;i<n;i++)
				nb.input(data.instance(i));
			System.out.println(" Instances input");
			System.out.println(" Output format retrieved");
//			nd=Filter.useFilter(data,nb);
//			System.out.println(" Filtered? num atts = "+nd.numAttributes()+" num inst = "+nd.numInstances()+" filter = "+nb);
			if(nb.batchFinished())
				System.out.println(" batch finished ");
			nd=nb.getOutputFormat();
			for(int i=0;i<n;i++)
			{
				temp=nb.output();
//				System.out.println(temp); 
				nd.add(temp);
			}
		}catch(Exception e)
		{
			System.out.println("Error in estimateMissing  = "+e.toString());
			nd=data;
			System.exit(0);
			
		}
		return nd;
		
		}
	
/**
 * Converts all the categorical variables to binary
 * 
 * NOTE dummy created for all values, so the matrix is not full rank
 * If a regression formulation required (e.g. 6 binarys for 7 attribute values)
 * call makeBinaryFullRank
 * @param data
 */
    public static Instances makeBinary(Instances data){
        NominalToBinary nb = new NominalToBinary();

        Instances nd;
        try{
            Instance temp;
            nb.setInputFormat(data);
            int n = data.numInstances();
            for(int i=0;i<n;i++)
                    nb.input(data.instance(i));
            nd=nb.getOutputFormat();
            for(int i=0;i<n;i++)
            {
                    temp=nb.output();
//				System.out.println(temp); 
                    nd.add(temp);
            }
        }catch(Exception e)
        {
            System.out.println("Error in NominalToBinary  = "+e.toString());
            nd=data;
            System.exit(0);
        }
        return nd;
    }
/**
 * generates white noise attributes and random classes
 * @param numAtts
 * @param numCases
 * @param numClasses
 * @return 
 */        
    public static Instances generateRandomProblem(int numAtts,int numCases, int numClasses){
        String name="Random"+numAtts+"_"+numCases+"_"+numClasses;
        ArrayList<Attribute> atts=new ArrayList<>(numAtts);
        for(int i=0;i<numAtts;i++){
            Attribute at=new Attribute("Rand"+i);//Assume defaults to numeric?
            atts.add(at);
        }
//Add class value
        ArrayList<String> vals=new ArrayList<>(numClasses);
        for(int i=0;i<numClasses;i++)
                vals.add(i+"");
        atts.add(new Attribute("Response",vals));
 //Add instances
        NormalDistribution norm=new NormalDistribution(0,1);
        Random rng=new Random();
        Instances data=new Instances(name,atts,numCases);
        data.setClassIndex(numAtts);
        for(int i=0;i<numCases;i++){
            Instance in= new DenseInstance(data.numAttributes());
           
            for(int j=0;j<numAtts;j++){
                double v=norm.simulate();
                in.setValue(j, v);
            }            
            //Class value
            double classV=rng.nextInt(numClasses);
            in.setValue(numAtts,classV);
            data.add(in);
        }
        return data;
    }
        
}
