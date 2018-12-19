/*
 this classifier does none of the transformations. It simply loads the 
 * problems it is told to. In build classifier it can load the CV weights or 
find them, by default through LOOCV. For classifiers, it defaults to a standard 
set with default parameters. Alternatively, you can set the classifiers and 
for certain types set the parameters through CV. 
 */
package weka.classifiers.meta.timeseriesensembles.depreciated;

import development.DataSets;
import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import timeseriesweka.classifiers.ensembles.SaveableEnsemble;
import utilities.ClassifierTools;
import weka.classifiers.*;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.kNN;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.rules.ZeroR;
import vector_classifiers.TunedRandomForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.*;
import utilities.SaveParameterInfo;

/**
 *
 * @author ajb: Described in draft paper
 * 
 * The Heterogeneous Ensemble of Standard Classification Algorithms (HESCA)
 * We use a Heterogeneous Ensemble of Standard Classification Algorithms (HESCA) that includes eight classifiers, two of which themselves are ensembles: $k$ Nearest Neighbour; Naive Bayes; C4.5 decision tree; Support Vector Machines with linear and quadratic basis function kernels; Random Forest (with 500 trees); Rotation Forest (with 50 trees); and a Bayesian network. These classifiers are chosen to give us a balance between probabilistic, instance based and tree based classifiers. A simple majority voting scheme is inappropriate for HESC; it would not capture the relative performance of classifier on any given dataset. Instead, each classifier is assigned a weight based on the ten fold cross validation training accuracy, and new data (after transformation) are classified with a vote weighted by cross validation accuracy. All the classifiers are the standard Weka implementations and with the exception of $k$-NN (where $k$ is set through cross validation at minimal cost), we do not optimise parameter settings or perform model selection for these classifiers. 
 */
public class HESCA_05_10_16 extends AbstractClassifier implements SaveParameterInfo, SaveableEnsemble {
//The McNemar test requires the actual predictions of each classifier. The others can be found directly
//from the CV accuracy.    
    protected Instances train;
    protected Classifier[] c;
    protected ArrayList<String> classifierNames;
    protected double[] cvAccs;
    double[] weights;
    boolean loadCVWeights=false;
    String cvFile="";
    public static int MAX_NOS_FOLDS=100;
    Random r= new Random();
    protected boolean setSeed=false;
    protected int seed;
    boolean saveTrain=false;
    boolean saveTest=false;
/* Train results are overwritten with each call to buildClassifier
    File opened on this path.   */    
    String trainDataResultsPath;
/*Test data must be explicitly closed in order to overwrite, otherwise it is kept
  open over different calls to classifyInstance   */
    OutFile testData;
    boolean memoryClean=true;
    @Override
    public void saveResults(String tr, String te){
        setCVPath(tr);
        saveTestPreds(te);
    }
   

    public enum WeightType{EQUAL,BEST,PROPORTIONAL,FIXED,SIGNIFICANT_BINOMIAL,SIGNIFICANT_MCNEMAR};
    WeightType w;
    public HESCA_05_10_16(){
        w=WeightType.PROPORTIONAL;
        classifierNames=new ArrayList<String>();
        c=setDefaultClassifiers(classifierNames);
        weights=new double[c.length];
        cvAccs=new double[c.length];
    }
    public HESCA_05_10_16(Classifier[] cl,ArrayList<String> names){
        w=WeightType.PROPORTIONAL;
        setClassifiers(cl,names);
        weights=new double[c.length];
        cvAccs=new double[c.length];
    }
    public void loadCVWeights(String file){
        loadCVWeights=true;
        cvFile=file;
        w= WeightType.FIXED;
    }
    public void setRandSeed(int s){
        r=new Random(s);
        setSeed=true;
        seed=s;
    }
    public void setWeightType(WeightType a){w=a;}
    public void setWeightType(String s){
        String str=s.toUpperCase();
        switch(str){
            case "EQUAL": case "EQ": case "E":
                w=WeightType.EQUAL;
                break;
            case "BEST": case "B":
                w=WeightType.BEST;
                break;
            case "PROPORTIONAL": case "PROP": case "P":
                w=WeightType.PROPORTIONAL;
                break;
 /*            case "SIGNIFICANT_BINOMIAL": case "SIGB": case "SIG": case "S":
                w=WeightType.SIGNIFICANT_BINOMIAL;
                break;
            case "SIGNIFICANT_MCNEMAR": case "SIGM": case "SM":
                w=WeightType.SIGNIFICANT_MCNEMAR;
                break;
 */      
            default:
                throw new UnsupportedOperationException("Weighting method "+str+" not supported yet.");       
        }
   
    }

    
    public void setCVPath(String s){
        saveTrain=true;
        trainDataResultsPath=s;
    }
    public String getParameters(){
        String p="";
        p+="NosClassifiers,"+c.length+",type,"+w;
        return p;
    }
    public void saveTestPreds(String s){
        saveTest=true;
        testData=new OutFile(s);
    }
    
    
/*The classifiers used are the WEKA [26] implementations
of k Nearest Neighbour (where k is set through cross
validation), Naive Bayes, C4.5 decision tree [27], Support
Vector Machines [28] with linear and quadratic basis
function kernels, Random Forest [29] (with 500 trees), Ro-
tation Forest [30] (with 50 trees), and a Bayesian network.
 */       
    final public Classifier[] setDefaultClassifiers(ArrayList<String> names){
            ArrayList<Classifier> classifiers=new ArrayList<>();
            kNN k=new kNN(100);
            k.setCrossValidate(true);
            k.normalise(false);
            k.setDistanceFunction(new EuclideanDistance());
            classifiers.add(k);
            names.add("NN");
            
            classifiers.add(new NaiveBayes());
            names.add("NB");
            J48 tree = new J48();
//            tree.setMinNumObj(5);
            classifiers.add(tree);
            names.add("C45");
            SMO svm=new SMO();
            PolyKernel kernel = new PolyKernel();
            kernel.setExponent(1);
            svm.setKernel(kernel);
            if(setSeed)
                svm.setRandomSeed(seed);
            classifiers.add(svm);
            names.add("SVML");
            svm=new SMO();
            kernel = new PolyKernel();
            kernel.setExponent(2);
            svm.setKernel(kernel);
            if(setSeed)
               svm.setRandomSeed(seed);
            classifiers.add(svm);
            names.add("SVMQ");
            RandomForest r=new TunedRandomForest();
            r.setNumTrees(500);
            if(setSeed)
               r.setSeed(seed);            
            classifiers.add(r);
            names.add("RandF");
            RotationForest rf=new RotationForest();
            rf.setNumIterations(50);
            if(setSeed)
               rf.setSeed(seed);
            classifiers.add(rf);
            names.add("RotF");
            BayesNet bn=new BayesNet();
            classifiers.add(bn);
            names.add("bayesNet");
            
            
            Classifier[] sc=new Classifier[classifiers.size()];
            for(int i=0;i<sc.length;i++)
                    sc[i]=classifiers.get(i);

            return sc;
    }


    final public void setClassifiers(Classifier[] cl,  ArrayList<String> names){
        c=cl;
        classifierNames=new ArrayList<>(names);
        weights=new double[c.length];
        cvAccs=new double[c.length];
    }

    public final double[] getWeights(){ return weights;}
    public final double[] getCVAccs(){ return cvAccs;}
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        train = data;
        
        if(data.numInstances()>500 || data.numAttributes()>500)
            MAX_NOS_FOLDS=10;
        OutFile of=null;
        if(saveTrain){
            of=new OutFile(trainDataResultsPath);
            for(String s:classifierNames)
                of.writeString(s+",");
            of.writeString("\n");
        }
//load CV, classifiers and parameter sets        
        if(w==WeightType.FIXED){
            if(!(new File(cvFile).exists())){
                //ERRROR, FILE DOES NOT EXIST
                System.out.println("ERROR: FILE "+cvFile+" DOES NOT EXIST .. its going to crash now ...");
            }
            InFile inf=new InFile(cvFile);
            inf.readLine(); //Header
            double sum=0;
            for(int i=0;i<c.length;i++){
                cvAccs[i]=inf.readDouble();
//                System.out.println(" CV ACC ="+cvAccs[i]);
                sum+=cvAccs[i];
            }
//Train classifiers           
            for(int i=0;i<c.length;i++)
                weights[i]=cvAccs[i]/sum;
            for(int i=0;i<c.length;i++)
                c[i].buildClassifier(train);
        }
//If using equal weighting, set cvAcc to 1 and weights to 1/nosClassifiers        
        else if(w==WeightType.EQUAL){
            for(int i=0;i<c.length;i++){
                cvAccs[i]=1;
                weights[i]=1/(double)c.length;
            }
//Final train            
            for(int i=0;i<c.length;i++)
                c[i].buildClassifier(train);
        }
//Else, find the cvAccs of the classifier through CV, then weight proportionally
//All weight types will require this        
        else{
            double sum=0;
            Evaluation eval;
            for(int i=0;i<c.length;i++){
/*Speed things up by using the OOB error rather than the CV error for random forest.                
    it should be possible to do this for rotation forest also            
                */
                if(c[i] instanceof TunedRandomForest){
                    c[i].buildClassifier(train);
                    cvAccs[i]=1-((TunedRandomForest)c[i]).measureOutOfBagError();
                    sum+=cvAccs[i];
//                    System.out.println(" OOB ="+(1-cvAccs[i]));
                }
                else{
                    eval=new Evaluation(train);
    //set the max number of folds to MAX_FOLDS or use LOOCV
                    int folds=10;
//Hugely memory intensive, so clean up if required
//The CV could be done much more efficiently in memory
/*There is an unusual problem with NB  and BN. If a subsample has a flat feature
(or almost flat) the discretisation fails. The hack to sort this is just to ignore
   these classifiers if it is the case
                    */
                    try{    
                        eval.crossValidateModel(c[i],train,folds,r);
                        cvAccs[i]=1-eval.errorRate();
                        c[i].buildClassifier(train);
                    }catch(Exception e){
                        System.out.println("Caught illegal argument exception "+e);
                        cvAccs[i]=0;    //Dont use the classifier
//Need something there just so it does not crash later
//But it will never be used                        
                        c[i]=new ZeroR();
                        c[i].buildClassifier(train);
                    }
/*                    catch(Error e){
// This is really bad practice, but WEKA ibk throws one of these for 
//reasons I'm not sure. However, it might completely screw up the system if an 
//out of memory error is thrown? DONT LEAVE THIS IN                        
                        System.out.println("Caught an ERROR!?! "+e);
                        cvAccs[i]=0;               
                        c[i]=new J48();
                        c[i].buildClassifier(train);
                    }
                        */                         
                    sum+=cvAccs[i];
                    if(memoryClean)
                        System.gc();
                }
            }
            for(int i=0;i<c.length;i++)
                weights[i]=cvAccs[i]/sum;
            if(saveTrain){
               for(int i=0;i<c.length;i++)
                   of.writeString(cvAccs[i]+",");
               of.writeString("\n"); 
//Need to store actuals and predicted here, but requires restructuring.
 //              for(int i=0;i<train.numInstances();i++)
 //                  of.writeString((int)train.instance(i).classValue()+",");
            }            
        }
        
 //If using Best find largest, set to one and others to zero.       
        if(w==WeightType.BEST){  
            int bestPos=0;
            for(int i=1;i<weights.length;i++){
                if(weights[i]>weights[bestPos])
                    bestPos=i;
            }
            for(int i=0;i<weights.length;i++){
                if(i==bestPos)
                    weights[i]=1;
                else
                    weights[i]=0;
            }
        }
    }
    
//Stores the current individual classifier predictions for the last call to
//classifyInstance
    protected double[] predictions;
    
    public double[] getPredictions(){ return predictions;}
    
    @Override
    public double[] distributionForInstance(Instance ins) throws Exception{
        predictions=new double[c.length];
        double[] preds=new double[ins.numClasses()];
        if(saveTest)
            testData.writeString(ins.classValue()+",");
        for(int i=0;i<c.length;i++){
            predictions[i]=c[i].classifyInstance(ins);
            if(saveTest)
                testData.writeString(predictions[i]+",");

//            System.out.println(" Classifier "+classifierNames.get(i)+" predicts class "+predictions[i]+" with weight "+weights[i]);
            preds[(int)predictions[i]]+=weights[i];
        }
        double sum=preds[0];
        for(int i=1;i<preds.length;i++){
            sum+=preds[i];
        }
        for(int i=0;i<preds.length;i++)
            preds[i]/=sum;
        if(saveTest){
            testData.writeString("\n");            
        }
        
        return preds;
    }
    public ArrayList<String> getNames(){ return classifierNames;}

    
    public static void debugTest(){
//Basic test of build classifer 
        String problem="ItalyPowerDemand";
        Instances train =ClassifierTools.loadData(DataSets.dropboxPath+problem+"\\"+problem+"_TRAIN");
        Instances test =ClassifierTools.loadData(DataSets.dropboxPath+problem+"\\"+problem+"_TEST");
        HESCA_05_10_16 we = new HESCA_05_10_16();
//Test equal weight and CV weight for small classifier set
        DecimalFormat df = new DecimalFormat("###.###");
        double a;
        try {
            Classifier[] c = new Classifier[3];
            ArrayList<String> names = new ArrayList<>();
            kNN k=new kNN(100);
            k.setCrossValidate(true);
            k.normalise(false);
            k.setDistanceFunction(new EuclideanDistance());
            c[0]=k;
            names.add("NN");
            c[1]=new NaiveBayes();
            names.add("NB");
            c[2]=new J48();
            names.add("C45");
            we.setClassifiers(c, names);
            we.setWeightType("EQUAL");
            we.buildClassifier(train);
            a=ClassifierTools.accuracy(test, we);
            System.out.println(" WE accuracy with equal weight="+a);
            we.setWeightType("Proportional");
            we.buildClassifier(train);
            double[] w=we.getWeights();
            double[] cv=we.getCVAccs();
            for(int i=0;i<w.length;i++)
                System.out.println("Weight ="+df.format(w[i])+" CV ="+df.format(cv[i]));
            a=ClassifierTools.accuracy(test, we);
            System.out.println(" WE accuracy with prop weight="+a);
            
        } catch (Exception ex) {
            Logger.getLogger(HESCA_05_10_16.class.getName()).log(Level.SEVERE, null, ex);
        }
//Test with standard classifiers
        try {
            we = new HESCA_05_10_16();
            we.setWeightType("Proportional");
            we.buildClassifier(train);
            double[] w=we.getWeights();
            double[] cv=we.getCVAccs();
            for(int i=0;i<w.length;i++)
                System.out.println("Weight ="+df.format(w[i])+" CV ="+df.format(cv[i]));
            a=ClassifierTools.accuracy(test, we);
            System.out.println(" WE accuracy with prop weight="+a);
            
            
        } catch (Exception ex) {
            Logger.getLogger(HESCA_05_10_16.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        
    }
//Test for the ensemble in the spectral data
 /*   public static void testSpectrum() throws Exception{
        Instances train=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\Power Spectrum Transformed TSC Problems\\PSItalyPowerDemand\\PSItalyPowerDemand_TRAIN");
        Instances test=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\Power Spectrum Transformed TSC Problems\\PSItalyPowerDemand\\PSItalyPowerDemand_TEST");
        HESCA w=new HESCA();
        w.buildClassifier(train);
        System.out.println(" Accuracy ="+ClassifierTools.accuracy(test, w));
    }
    
*/    

    public static void testFileSave() throws Exception{
        Instances train= ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\ItalyPowerDemand\\ItalyPowerDemand_TRAIN");
        Instances test= ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\ItalyPowerDemand\\ItalyPowerDemand_TEST");
        HESCA_05_10_16 we= new HESCA_05_10_16();
        we.saveTestPreds("C:\\Users\\ajb\\Dropbox\\TSC Problems\\ItalyPowerDemand\\TestPreds.csv");
        we.setCVPath("C:\\Users\\ajb\\Dropbox\\TSC Problems\\ItalyPowerDemand\\TrainCV.csv");
        we.buildClassifier(train);
        for(Instance ins:test){
            we.classifyInstance(ins);
        }

    }

    public static void main(String[] args) throws Exception{
        testFileSave();
//            debugTest();
    }
}
