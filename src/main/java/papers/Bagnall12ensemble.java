/**
 * Code to reproduce the results in the paper
 * @inproceedings{bagnall12ensemble,
	Author = {A. Bagnall and L. Davis and J. Hills  and J. Lines},
	Title ="Transformation Based Ensembles for Time Series Classification",
	Booktitle ="Proceedings of the 12th {SIAM} International Conference on Data Mining (SDM)",
    pages="307--319",
	Year = {2012}
}
 * 
 */

package papers;

import timeseriesweka.filters.PowerSpectrum;
import timeseriesweka.filters.ACF;
import fileIO.InFile;
import fileIO.OutFile;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Random;
import utilities.ClassifierTools;
import weka.attributeSelection.PrincipalComponents;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import timeseriesweka.classifiers.FastDTW_1NN;
import weka.classifiers.lazy.kNN;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.RotationForest;
import timeseriesweka.classifiers.ensembles.TransformEnsembles;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.filters.NormalizeCase;
import development.*;
import weka.classifiers.AbstractClassifier;


public class Bagnall12ensemble {
	static String resultPath="C:\\Users\\ajb\\Dropbox\\Results\\Ensembles";
		public static String[] sdm2012fileNames={	//Number of train,test cases,length,classes
				"Adiac",//390,391,176,37
				"ARSim",
				"Beef", //30,30,470,5
				"CBF",//30,900,128,3
				"Lighting2",//60,61,637,2
				"Lighting7",//70,73,319,7
				"ECG200",//100,100,96,2
				"FaceFour",//24,88,350,4
				"fiftywords",//450,455,270,50
				"fish",//175,175,463,7
				"GunPoint",//50,150,150,2
				"OSULeaf", //200,242,427,6
				"SwedishLeaf", //500,625,128,15
				"SyntheticControl", //300,300,60,6
				"Trace",//100,100,275,4
				"TwoPatterns", //1000,4000,128,4
				"wafer",//1000,6174,152,2
				"yoga",//300,3000,426,2
				"FaceAll",//560,1690,131,14
				//Index 18, after this the data has not been normalised.
				"Coffee", //28,28,286,2
				"OliveOil",
				"Earthquakes",
				"HandOutlines",//1000,300,2790
				"FordA",
				"FordB",
				"ElectricDevices",
		};
	

	public static boolean normalise(String fileName){
		if(fileName.equals("FordA")||fileName.equals("FordB")||fileName.equals("OliveOil")||fileName.equals("Beef")||fileName.equals("Coffee")||fileName.equals("Earthquakes"))
			return true;
		return false;
	}
	
	
	
	public static Classifier[] getFilters(ArrayList<String> names){
		ArrayList<Classifier> sc2=new ArrayList<Classifier>();
		Classifier c;
		//1. Basic 1-NN Euclidean distance
		kNN c1;
		for(double i=1;i>0;i-=0.1){
			c1=new kNN(1);
			c1.setFilterAttributes(true);
			c1.setProportion(i);
			sc2.add(c1);
			names.add("(1NN"+i+")");
		}
		Classifier[] sc=new Classifier[sc2.size()];
		for(int i=0;i<sc.length;i++)
			sc[i]=sc2.get(i);
		return sc;
	}

	
	public static Classifier[] setSingleClassifiers(ArrayList<String> names){
		ArrayList<Classifier> sc2=new ArrayList<Classifier>();
		sc2.add(new kNN(1));
		names.add("NN");
		Classifier c;
		c=new FastDTW_1NN();
		((FastDTW_1NN)c).optimiseWindow(false);
		
		sc2.add(c);
		names.add("NNDTW");
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
		c=new SMO();
		RBFKernel kernel2 = new RBFKernel();
		((SMO)c).setKernel(kernel2);
		sc2.add(c);
		names.add("SVMR");
		c=new RandomForest();
		((RandomForest)c).setNumTrees(30);
		sc2.add(c);
		names.add("RandF30");
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

	
	public static Classifier[] setNNClassifiers(ArrayList<String> names){
		ArrayList<Classifier> sc2=new ArrayList<Classifier>();
		Classifier c;
		//1. Basic 1-NN Euclidean distance
		kNN c1=new kNN(1);
		sc2.add(c1);
		names.add("(1NN)");
		//2. k-NN, k set through LOOCV
		kNN c2=new kNN(50);
		c2.setCrossValidate(true);
		sc2.add(c2);
		names.add("(kNN)");
		
		//3. 1-NN Filtered 50%
		kNN c3=new kNN(1);
		c3.setFilterAttributes(true);
		c3.setProportion(0.5);
		sc2.add(c3);
		names.add("(1NN-50\\% Filter");
		//Bagging with 20 base classifiers
		int bagPercent=50;
		c=new Bagging();
		((Bagging)c).setClassifier(new kNN(1));
		((Bagging)c).setNumIterations(20);
		((Bagging)c).setBagSizePercent(bagPercent);
		names.add("Bagging,"+bagPercent+"%,20 1NN");
		sc2.add(c);
		//Bagging with 100 base classifiers
		c=new Bagging();
		((Bagging)c).setClassifier(new kNN(1));
		((Bagging)c).setNumIterations(50);
		((Bagging)c).setBagSizePercent(66);
		names.add("Bagging,"+bagPercent+"%,100 1NN");
		sc2.add(c);
		//Boosting with 20 base
		c=new AdaBoostM1();
		((AdaBoostM1)c).setClassifier(new kNN(1));
		((AdaBoostM1)c).setNumIterations(20);
		((AdaBoostM1)c).setUseResampling(true);
		sc2.add(c);
		names.add("Boosting 20 1NN");
		Classifier[] sc=new Classifier[sc2.size()];
		for(int i=0;i<sc.length;i++)
			sc[i]=sc2.get(i);
		System.out.print("Testing NN Classifiers: ");
		for(String s:names)
			System.out.print(s+",");

		return sc;
	}



	/**
	 * Generates the results for Table 2: Classifier Comparison in the time domain (argument "SingleClassifiers")
	 * and for Figure 8, NN Variants(argument "NN_EuclidClassifiers"
	 * @param tableName
	 */
	public static void timeDomain(String tableName){
		OutFile of=new OutFile(resultPath+"table"+tableName+".csv");
		ClassifierTools.ResultsStats[] stats;
		ArrayList<String> names=new ArrayList<String>();
		Classifier[] sc=null;
		if(tableName.equals("SingleClassifiers")){
			System.out.println("SINGLE CLASSIFIERS");
			of.writeLine("SINGLE CLASSIFIERS,  results in Time Domain");
			sc=setSingleClassifiers(names);
		}
		else if(tableName.equals("NN_EuclidClassifiers")){
			System.out.println("NN Euclidean CLASSIFIER VARIANTS");
			of.writeLine("NN CLASSIFIERS,  results in Time Domain");
			sc=setNNClassifiers(names);
		}
		else{
			System.out.println("ERROR: Unknown classifier generation name");
			System.exit(0);
		}
		of.writeString(",,");
		for(String s:names){
			of.writeString(s+",");	
		}
		
	of.writeLine("\n");
		try{
			for(int i=0;i<sdm2012fileNames.length;i++)
			{
//Load default test train split
				Instances test=ClassifierTools.loadData(DataSets.dropboxPath+sdm2012fileNames[i]+"\\"+sdm2012fileNames[i]+"_TEST");
				Instances train=ClassifierTools.loadData(DataSets.dropboxPath+sdm2012fileNames[i]+"\\"+sdm2012fileNames[i]+"_TRAIN");			

/*				//Resampling for DTW
				if(tooLarge(DataSets.tscProblems85[i])){
					System.out.println("Sampling  ....");
					if(DataSets.tscProblems85[i].equals("Earthquakes"))
						train=sample(train,0.3);
					else{
						train=sample(train,0.1);
						test=sample(test,0.3);
					}
				}
*/				
//Normalise if necessary.
				
				if(normalise(DataSets.tscProblems85[i])){
					System.out.println("Standardising "+DataSets.tscProblems85[i]);
					NormalizeCase nc=new NormalizeCase();
					train=nc.process(train);
					test=nc.process(test);
				}

				//Reinitialise the classifier each time for safety sake.
				if(tableName.equals("SingleClassifiers"))
					sc=setSingleClassifiers(names);
				else if(tableName.equals("NN_EuclidClassifiers"))
					sc=setNNClassifiers(names);
					
				//Set folds. If 1 then it does the  test/train split defined by the two files			
				int folds=setNosFolds(test,train);			
				of.writeString("\n"+DataSets.tscProblems85[i]+","+folds+",");
				System.out.println("Train size = "+train.numInstances()+" Test size ="+test.numInstances()+" folds ="+folds);
				System.out.println(DataSets.tscProblems85[i]);
				System.out.println("************************************");

				//Returns an array of stats, only using accuracy at present.
				stats=ClassifierTools.evalClassifiers(test,train,folds,sc);
				for(int j=0;j<stats.length;j++){
					of.writeString(stats[j].accuracy+",");
					System.out.println("\t"+names.get(j)+" error ="+(1-stats[j].accuracy));
				}
			}
		}catch(Exception e){
			System.out.println("Exception = "+e);
			e.printStackTrace();
		}
			
	}

/** 
 * Generates the results for the first half of Table 3: Comparison of transforms
 */
	public static void basicDataTransforms(String baseClassifier){
		DecimalFormat dc= new DecimalFormat("###.###");
		OutFile of=new OutFile(resultPath+baseClassifier+"BasicDataTransforms.csv");
		ClassifierTools.ResultsStats[] stats;
		System.out.println("NEAREST NEIGHBOUR CLASSIFIERS");
		of.writeLine("NEAREST NEIGHBOUR CLASSIFIERS, 10 fold cross validation results");
		of.writeLine(",TimeDomain,PowerSpectrumDomain,ACFDomain,PCADomain");
		of.writeLine(",1-NN");
		String[] files=DataSets.tscProblems85;
		Classifier base=null;
		if(baseClassifier.equals("1NN"))
			base=new kNN(1);
		else if(baseClassifier.equals("DTW"))
			base=new FastDTW_1NN();
		else if(baseClassifier.equals("RotationForest"))
			base=new RotationForest();
		else if(baseClassifier.equals("RandomForest")){
			base=new RandomForest();
			((RandomForest)base).setNumTrees(30);
		}
		else{
			System.out.println("Classifier Not Included, exiting");
			System.exit(0);
		}
			
		
		try{
			for(int i=0;i<files.length;i++)
			{
				Instances test=ClassifierTools.loadData(DataSets.dropboxPath+files[i]+"\\"+files[i]+"_TEST");
				Instances train=ClassifierTools.loadData(DataSets.dropboxPath+files[i]+"\\"+files[i]+"_TRAIN");			
				of.writeString("\n"+DataSets.tscProblems85[i]+",");
				System.out.println("\n"+DataSets.tscProblems85[i]+",");
				//Set folds
				int folds=setNosFolds(test,train);
				Classifier[] sc= new Classifier[1];
				Instances timeTrain, timeTest;
				if(normalise(DataSets.tscProblems85[i])){
					System.out.println("Standardising "+DataSets.tscProblems85[i]);
					NormalizeCase nc=new NormalizeCase();
					timeTrain=new Instances(train);
					timeTest=new Instances(test);
					nc.process(timeTrain);
					nc.process(timeTest);
				}				
				else{
					timeTrain=train;
					timeTest=test;
				}
				//Time domain: no need as we already have these results			
				System.out.println("******************Time Domain******************");
  
				sc[0]=AbstractClassifier.makeCopy(base);
//				sc[1]=new FastDTW_1NN(1);
				stats=ClassifierTools.evalClassifiers(timeTest,timeTrain,folds,sc);
				for(int j=0;j<stats.length;j++){
					of.writeString(stats[j].accuracy+",");
					System.out.print("\n \t TIME: "+dc.format(stats[j].accuracy)+",");
				}			

				System.out.println("******************Power Spectrum Domain******************");
				PowerSpectrum ps=new PowerSpectrum();
				Instances psTrain=ps.process(train);
				Instances psTest=ps.process(test);
				psTrain.deleteAttributeAt(0);
				psTest.deleteAttributeAt(0);
/* Delete the duplicate half of the spectrum */
 				int atts=(psTrain.numAttributes()-1)/2-2;
				for(int j=0;j<atts;j++){
					psTrain.deleteAttributeAt(psTrain.numAttributes()-2);
					psTest.deleteAttributeAt(psTest.numAttributes()-2);
				}
				sc[0]=AbstractClassifier.makeCopy(base);
				stats=ClassifierTools.evalClassifiers(psTest,psTrain,folds,sc);
//Remove the last 50% of the coefficients.
				for(int j=0;j<stats.length;j++){
					of.writeString(stats[j].accuracy+",");
					System.out.print("\n\t SPECTRUM: "+dc.format(stats[j].accuracy)+",");
				}
				System.out.println("\n******************ACF Domain******************");
				ACF acf=new ACF(); 
				acf.setMaxLag(train.numAttributes()-(int)(train.numAttributes()*.1));
				sc[0]=AbstractClassifier.makeCopy(base);
				Instances acfTrain=acf.process(train);
				Instances acfTest=acf.process(test);
/*				atts=(acfTrain.numAttributes()-1)/2;
				for(int j=0;j<atts;j++){
					acfTrain.deleteAttributeAt(acfTrain.numAttributes()-2);
					acfTest.deleteAttributeAt(acfTest.numAttributes()-2);
				}
*/				stats=ClassifierTools.evalClassifiers(acfTest,acfTrain,folds,sc);
				for(int j=0;j<stats.length;j++){
					of.writeString(stats[j].accuracy+",");
					System.out.print("\n\t ACF: "+dc.format(stats[j].accuracy)+",");
				}
				System.out.println("\n******************PCA Domain******************");
				PrincipalComponents pca=new PrincipalComponents (); 
				sc[0]=AbstractClassifier.makeCopy(base);
				pca.buildEvaluator(train);
				Instances pcaTrain=pca.transformedData(train);
				Instances pcaTest=pca.transformedData(test);/*
				atts=(pcaTrain.numAttributes()-1)/2;
				for(int j=0;j<atts;j++){
					pcaTrain.deleteAttributeAt(pcaTrain.numAttributes()-2);
					pcaTest.deleteAttributeAt(pcaTest.numAttributes()-2);
				}
*/				stats=ClassifierTools.evalClassifiers(pcaTest,pcaTrain,folds,sc);
				for(int j=0;j<stats.length;j++){
					of.writeString(stats[j].accuracy+",");
					System.out.print("\n\t Pca: "+dc.format(stats[j].accuracy)+",");
				}
			}
		}catch(Exception e){System.out.println("Exception ="+e);e.printStackTrace();System.exit(0);}
	}	
		
	/** 
	 * Generates the results for the second half of Table 3: Comparison of ensemblese
	 * This is also used to generate pairwise comparisons for Figure 10
	 */
	public static void ensembleTransforms(String baseClassifier){
		DecimalFormat dc= new DecimalFormat("###.###");
		OutFile of=new OutFile(resultPath+baseClassifier+"EnsembleTransforms.csv");
		OutFile of2=new OutFile(resultPath+baseClassifier+"EnsembleWeights.csv");
		ClassifierTools.ResultsStats[] stats;
		System.out.println("ENSEMBLECLASSIFIERS");
		of.writeLine(baseClassifier+",CombinedEqual,CombinedBest,CombinedWeighted,CombinedStep");
		String[] files=DataSets.tscProblems85;
		Classifier base=null;
		if(baseClassifier.equals("1NN"))
			base=new kNN(1);
		else if(baseClassifier.equals("DTW"))
			base=new FastDTW_1NN();
		else if(baseClassifier.equals("RotationForest"))
			base=new RotationForest();
		else if(baseClassifier.equals("RandomForest")){
			base=new RandomForest();
			((RandomForest)base).setNumTrees(30);
		}
		else if(baseClassifier.equals("C4.5")){
			base=new J48();
		}
		else if(baseClassifier.equals("NB")){
			base=new NaiveBayes();
		}
		else if(baseClassifier.equals("SVMO")){
			base=new SMO();
			PolyKernel kernel = new PolyKernel();
			kernel.setExponent(2);
			((SMO)base).setKernel(kernel);
		}
		else if(baseClassifier.equals("SVML")){
			base=new SMO();
			PolyKernel kernel = new PolyKernel();
			kernel.setExponent(1);
			((SMO)base).setKernel(kernel);
		}
		else{
			System.out.println("Classifier Not Included, exiting");
			System.exit(0);
		}
			
		
		try{
			for(int i=0;i<files.length;i++)
			{
				Instances test=ClassifierTools.loadData(DataSets.dropboxPath+files[i]+"\\"+files[i]+"_TEST");
				Instances train=ClassifierTools.loadData(DataSets.dropboxPath+files[i]+"\\"+files[i]+"_TRAIN");			
				of.writeString("\n"+DataSets.tscProblems85[i]+",");
				System.out.println("\n"+DataSets.tscProblems85[i]+",");
				//Set folds
				int folds=setNosFolds(test,train);
				Classifier[] sc= new Classifier[1];
				Instances timeTrain, timeTest;
				if(normalise(DataSets.tscProblems85[i])){
					System.out.println("Standardising "+DataSets.tscProblems85[i]);
					NormalizeCase nc=new NormalizeCase();
					timeTrain=new Instances(train);
					timeTest=new Instances(test);
					nc.process(timeTrain);
					nc.process(timeTest);
				}				
				else{
					timeTrain=train;
					timeTest=test;
				}
				sc[0]=new TransformEnsembles();
		
					System.out.println("\n******************Combined Equal******************");

				((TransformEnsembles)sc[0]).setBaseClassifier(AbstractClassifier.makeCopy(base));
				((TransformEnsembles)sc[0]).setWeightType(TransformEnsembles.WeightType.EQUAL);
				stats=ClassifierTools.evalClassifiers(test,train,folds,sc);
				for(int j=0;j<stats.length;j++){
					of.writeString(stats[j].accuracy+",");
					System.out.print("\n\t ENSEMBLE_EQUAL: "+dc.format(stats[j].accuracy)+",");
				}
				System.out.println("\n******************Combined Best******************");
				((TransformEnsembles)sc[0]).setBaseClassifier(AbstractClassifier.makeCopy(base));
				((TransformEnsembles)sc[0]).setWeightType(TransformEnsembles.WeightType.BEST);
				((TransformEnsembles)sc[0]).rebuildClassifier(false);
				((TransformEnsembles)sc[0]).findWeights();
				stats=ClassifierTools.evalClassifiers(test,train,folds,sc);
				for(int j=0;j<stats.length;j++){
					of.writeString(stats[j].accuracy+",");
					System.out.print("\n\t ENSEMBLE_BEST: "+dc.format(stats[j].accuracy)+",");
				}
				System.out.println("\n******************Combined Weighted******************");
				((TransformEnsembles)sc[0]).setBaseClassifier(AbstractClassifier.makeCopy(base));
				((TransformEnsembles)sc[0]).setWeightType(TransformEnsembles.WeightType.CV);
				((TransformEnsembles)sc[0]).rebuildClassifier(false);
				((TransformEnsembles)sc[0]).findWeights();
				stats=ClassifierTools.evalClassifiers(test,train,folds,sc);
				for(int j=0;j<stats.length;j++){
					of.writeString(stats[j].accuracy+",");
					System.out.print("\n\t ENSEMBLE_WEIGHT: "+dc.format(stats[j].accuracy)+",");
				}
				System.out.println("\n******************Combined STEP******************");
				((TransformEnsembles)sc[0]).setBaseClassifier(AbstractClassifier.makeCopy(base));
				((TransformEnsembles)sc[0]).setWeightType(TransformEnsembles.WeightType.STEP);
				((TransformEnsembles)sc[0]).rebuildClassifier(false);
				((TransformEnsembles)sc[0]).findWeights();
				stats=ClassifierTools.evalClassifiers(test,train,folds,sc);
				for(int j=0;j<stats.length;j++){
					of.writeString(stats[j].accuracy+",");
					System.out.print("\n\t ENSEMBLE_WEIGHT: "+dc.format(stats[j].accuracy)+",");
				}
				String w=((TransformEnsembles)sc[0]).getWeights();
				String w2=((TransformEnsembles)sc[0]).getCV();
				of2.writeLine(DataSets.tscProblems85[i]+","+w+","+w2);
			}
		}catch(Exception e){System.out.println("Exception ="+e);e.printStackTrace();System.exit(0);}
	}	
	



	

	

	/** Outputs a latex table with ranks, plus a matlab formatted table of Errors plus a spearate names file
	 * 
	 * @param str
	 * @param dest
	 */
	public static void formatRankTable(String str, String dest){
			InFile in=new InFile(str+"Acc.csv");
			InFile in2=new InFile(str+"Rank.csv");
			int lines=in.countLines();
			in=new InFile(str+"Acc.csv");
			String names=in.readLine();
			in2.readLine();			
			String[] classifiers=names.split(",");
			int nosClassifiers=classifiers.length-1;
			lines--;
			System.out.println("FILE PATH ="+str);
			System.out.println("nos problems ="+lines+" nos classifiers="+nosClassifiers);
			String[] problems=new String[lines];
			double[][] acc=new double[lines][nosClassifiers];
			double[][] ranks=new double[lines][nosClassifiers];
			for(int i=0;i<lines;i++){
				problems[i]=in.readString();
				System.out.print("Problem ="+problems[i]);
				in2.readString();
				for(int j=0;j<nosClassifiers;j++){
					acc[i][j]=in.readDouble();
				    ranks[i][j]=in2.readDouble();
					System.out.print(" "+classifiers[j]+" "+acc[i][j]+" ("+ranks[i][j]+")");				    
				}
				System.out.print(" \n");				    
			}
//Header
			OutFile of=new OutFile(dest);
			of.writeLine("\\begin{table*}[!ht]\n  \\scriptsize \n \\begin{tabular}{");
			for(int i=0;i<nosClassifiers;i++)
				of.writeString("c|");
			of.writeLine("c} \\hline \n Data Set\t&");
			for(int i=0;i<nosClassifiers-1;i++)
				of.writeString(classifiers[i+1]+"\t&");
			of.writeLine(classifiers[nosClassifiers]+"\\\\ \\hline");
			DecimalFormat df=new DecimalFormat("##.####");
			for(int i=0;i<problems.length;i++){
				of.writeString(problems[i]+"\t & ");
				for(int j=0;j<nosClassifiers;j++){
//If top ranked put in bold
					if(ranks[i][j]<2.0)
						of.writeString("{\\bf ");
					of.writeString(df.format(1-acc[i][j]));
					if(ranks[i][j]*10 ==((int)ranks[i][j])*10) //whole integer
						of.writeString("("+(int)ranks[i][j]+")");
					else		
						of.writeString("("+ranks[i][j]+")");
					if(ranks[i][j]<2.0)
						of.writeString("}");
					if(j==nosClassifiers-1){
						if(i<problems.length-1)
							of.writeString("\\\\ \n");
						else
							of.writeString("\\\\ \\hline \n");
					}
					else	
						of.writeString("\t & ");
				}
				
			}
			of.writeString("Mean Rank \t & ");
//Find Average Ranks and relevant stats
			double[] meanRanks=new double[nosClassifiers];
			double rSS=0;
			for(int i=0;i<problems.length;i++){
				for(int j=0;j<nosClassifiers;j++)
					meanRanks[j]+=ranks[i][j];
			}			
			for(int j=0;j<nosClassifiers;j++){
				meanRanks[j]/=problems.length;
				rSS+=meanRanks[j]*meanRanks[j];
				if(j<nosClassifiers-1)
					of.writeString(df.format(meanRanks[j])+" \t & ");
				else
					of.writeString(df.format(meanRanks[j])+" \\\\ \\hline ");
			}
			//Pairwise test statistics using the first as the control. 
			
	//		Q= \frac{12n}{k(k+1)} \cdot \left[ \sum_{j=1}^k\bar{r}^2_j-\frac{k(k+1)^2}{4}\right
			double n=problems.length;
			double k=nosClassifiers;
			double Q=12*n/(k*(k+1));
			Q*=(rSS-k*(k+1)*(k+1)/4);
			double F=(n-1)*Q;
			F/=n*(k-1)-Q;
			double[] zStat=new double[nosClassifiers];
			for(int j=1;j<nosClassifiers;j++)
				zStat[j]=(meanRanks[0]-meanRanks[j])/Math.sqrt((k*(k+1))/(6*n));
			of.writeString("\n &");
					for(int j=0;j<nosClassifiers;j++){
				if(j<nosClassifiers-1)
					of.writeString(df.format(zStat[j])+" \t & ");
				else
					of.writeString(df.format(zStat[j])+" \\\\ \\hline ");
			}
				of.writeString("\n");
//Footer
			of.writeLine("\\end{tabular} \n \\caption{Statistics: Q Stat="+df.format(Q)+" F Stat="+df.format(F)+"  CD="+"     }\n \\label{}\n \\end{table*}");
//Matlab format for CD graph
			OutFile o2=new OutFile(str+"Error.csv");
			OutFile o3=new OutFile(str+"Names.csv");
			o3.writeString("{");
			for(int i=1;i<classifiers.length;i++)
				o3.writeString("'"+classifiers[i]+"' ");
			o3.writeString("}");
			for(int i=0;i<problems.length;i++){
				for(int j=0;j<nosClassifiers;j++){
					if(j<nosClassifiers-1)
						o2.writeString(df.format((1-acc[i][j]))+",");
					else
						o2.writeString(df.format((1-acc[i][j]))+"\n");
				}	
			}	
	}
	public static void summariseData(String path)
	{
		OutFile of =new OutFile(path);
		try{
			for(int i=0;i<DataSets.tscProblems85.length;i++)
			{
//Load default test train split
				Instances test=ClassifierTools.loadData(DataSets.dropboxPath+DataSets.tscProblems85[i]+"\\"+DataSets.tscProblems85[i]+"_TEST");
				Instances train=ClassifierTools.loadData(DataSets.dropboxPath+DataSets.tscProblems85[i]+"\\"+DataSets.tscProblems85[i]+"_TRAIN");			
				of.writeString(DataSets.tscProblems85[i]+","+train.numInstances()+","+test.numInstances());
				of.writeString(","+(train.numAttributes()-1)+","+train.numClasses());
				double[] classDist=new double[train.numClasses()];
				for(int j=0;j<train.numInstances();j++)
					classDist[(int)train.instance(j).classValue()]++;
				for(int j=0;j<train.numClasses();j++)
					of.writeString(","+(classDist[j]/train.numInstances()));
				of.writeString(",,");
				classDist=new double[test.numClasses()];
				for(int j=0;j<test.numInstances();j++)
					classDist[(int)test.instance(j).classValue()]++;
				for(int j=0;j<test.numClasses();j++)
					of.writeString(","+(classDist[j]/test.numInstances()));
				of.writeString("\n");
				
			}
		}catch(Exception e){
			e.printStackTrace();
			System.exit(0);
		}
		
	}
		//Sanity check to confirm NN and DTW works

	
	public static void main(String[] args){
            ensembleTransforms("1NN");
            
//		formatRankTable("C:\\Users\\ajb\\Dropbox\\Results\\RSC\\NonSubspace","C:\\Users\\ajb\\Dropbox\\Results\\RSC\\NonSubspaceTable.csv");
//basic_recreateEamonnResults();

//		summariseData("C:\\Research\\Data\\Time Series Data\\Time Series Classification\\Summary.csv");
//Table 1: Compare alternative classifiers on the raw data	
	//	timeDomain("SingleClassifiers");
//Table 2: Compare alternative 1-NN Euclid classifiers on the raw data	
//		timeDomain("NN_EuclidClassifiers", new kNN(1));
		//Table 2: Compare alternative 1-NN DTW classifiers on the raw data	
//		timeDomain("NN_EuclidClassifiers", new FastDTW_1NN(1));
//		dataTransforms("DTW");
//		basicDataTransforms("1NN");		
//		ensembleTransforms("1NN");
//		ensembleTransforms("DTW");
//		ensembleTransforms("NB");
//		ensembleTransforms("C4.5");
//		ensembleTransforms("SVML");
//		ensembleTransforms("SVMO");
/*		ensembleTransforms("RandomForest");
		ensembleTransforms("RotationForest");
*/
		
		//		table2_NN_Combinations();
//		table3_1NN_DataTransforms();
		//		table3_1NN_Ensembles();
//		table4_My_Ensembles();	
//		SMO_Variants();
//		testNormalisation();
//		formatRankTable("C:\\Research\\Results\\TSC Results\\NNComparison","C:\\Research\\Results\\TSC Results\\NNComparisonLatex.csv");
//		formatRankTable("C:\\Research\\Results\\TSC Results\\EnsembleTransformComparison","C:\\Research\\Results\\TSC Results\\EnsembleTransformComparisonLatex.csv");
//		testECG();
	}

	
	public static void table4_My_Ensembles(){
		OutFile of=new OutFile(resultPath+"NewEnsemblesComparison.csv");
		int seed=100;
		ClassifierTools.ResultsStats stats;
		System.out.println("Ensemble on several transformations");
		try{
			for(int i=0;i<DataSets.tscProblems85.length;i++)
			{
				Instances test=ClassifierTools.loadData(DataSets.dropboxPath+DataSets.tscProblems85[i]+"\\"+DataSets.tscProblems85[i]+"_TEST");
				Instances train=ClassifierTools.loadData(DataSets.dropboxPath+DataSets.tscProblems85[i]+"\\"+DataSets.tscProblems85[i]+"_TRAIN");			
				of.writeString("\n"+DataSets.tscProblems85[i]+",");
				System.out.println(DataSets.tscProblems85[i]+",");
							//Time domain			
				//Set folds
				int folds=setNosFolds(test,train);
				TransformEnsembles te=new TransformEnsembles();
				double testAccuracy=0;
				double[][]  preds;

				if(folds>1){	// Combine the two files
					Instances full=new Instances(train);//Instances.mergeInstances(train, test);
					for(int j=0;j<test.numInstances();j++)
						full.add(test.instance(j));
		            Random rand = new Random(seed);
//					System.out.print("\t cases ="+full.numInstances());
		            full.randomize(rand);
					preds=ClassifierTools.crossValidation(te,full,folds);
					testAccuracy=preds[0][0];
				}
				else{
					te.buildClassifier(train);
					testAccuracy=ClassifierTools.accuracy(test,te);
				}
				System.out.println("\t : "+testAccuracy);
				of.writeString(testAccuracy+",");
				
			}
		}catch(Exception e){
			System.out.println("Exception = "+e);
			e.printStackTrace();
			System.exit(0);
		}
			
	}

	public static boolean tooLarge(String name){
		if(name.equals("FordA")||name.equals("FordB")||name.equals("HandOutlines")||name.equals("ElectricDevices")||name.equals("ARSim")||name.equals("Earthquakes"))
			return true;
		return false;
	}
	public static Instances sample(Instances data, double prop){
		if(prop<0||prop>1) return null;
		Instances newD=new Instances(data);
		newD.randomize(new Random());
		int size=(int)(prop*newD.numInstances());
		for(int i=size+1;i<data.numInstances();i++)
			newD.delete(size);
		return newD;
	}
	public static int setNosFolds(Instances test, Instances train){
		//Set to 1 to reproduce test/train results in line with Keogh website
				int folds	=1;	//train.numInstances();	
				
				//test.numInstances()+train.numInstances();
				if(folds>100)
					if(folds>1000)
						folds=1;
					else if(folds>500)
						folds=10;
				return folds;
			}	
}
