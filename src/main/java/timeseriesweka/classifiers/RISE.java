package timeseriesweka.classifiers;
/**
 * Development code for RISE
 * 1. set number of trees to max(500,m)
 * 2. Set the first tree to the full interval
 * 2. Randomly select the interval length and start point for each other tree *
 * 3. Find the PS, ACF, PACF and AR features
 * 3. Build each tree.

 **/ 

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
import java.util.ArrayList;
import java.util.Random;
import utilities.ClassifierTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomTree;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import timeseriesweka.filters.ACF;
import timeseriesweka.filters.PowerSpectrum;
import timeseriesweka.classifiers.SubSampleTrain;
import weka.core.Capabilities;

/**
 <!-- globalinfo-start -->
 * Random Interval Spectral Ensemble
 *
 * This implementation is the base RISE
 * Overview: Input n series length m
 * for each tree
 *      sample interval of random size (minimum set to 16)
 *      transform interval into ACF, PS, AR and PACF features
 *      build tree on concatenated features
 * ensemble the trees with majority vote
 <!-- globalinfo-end -->
 <!-- technical-bibtex-start -->
 * Bibtex
 * <pre>
 *   @article{lines2018time,
 *   title={Time series classification with HIVE-COTE: The hierarchical vote collective of transformation-based ensembles},
 *   author={Lines, Jason and Taylor, Sarah and Bagnall, Anthony},
 *   journal={ACM Transactions on Knowledge Discovery from Data (TKDD)},
 *   volume={12},
 *   number={5},
 *   pages={52},
 *   year={2018},
 *   publisher={ACM}
 *   }
 * </pre>
 <!-- technical-bibtex-end -->
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -T
 *  set number of trees.</pre>
 * 
 * <pre> -F
 *  set number of features.</pre>
 <!-- options-end -->
 * @author Tony Bagnall
 * @date Some time in 2017
 **/


public class RISE extends AbstractClassifierWithTrainingInfo implements SaveParameterInfo, SubSampleTrain{
    /** Default to a random tree */
    Classifier baseClassifierTemplate=new RandomTree();
    /** Ensemble base classifiers */    
    Classifier[] baseClassifiers;
    /** Ensemble size */    
    int numBaseClassifiers=500;
    /** Random Intervals for the transform. INTERVAL BOUNDS ARE INCLUSIVE  */  
    int[] startPoints;
    int[] endPoints;
    /** Minimum sizer of all intervals */    
    private int minInterval=16;
    
    /** Power Spectrum transformer, probably dont need to store this here  */   
    private PowerSpectrum ps=new PowerSpectrum();
    /**Can seed for reproducibility */
    private Random rand;
    private int seed=0;
    private boolean setSeed=false;
    
    /** If we are estimating the CV, it is possible to sample fewer elements. **/
    //Really should try bagging this!    
    private boolean subSample=false;
    private double sampleProp=1;
    
    /**
     * This interface is not formalised and needs to be considered in the next
     * review
     * @param prop
     * @param s 
     */    
    @Override
    public void subSampleTrain(double prop, int s){
        subSample=true;
        sampleProp=prop;
        seed=s;
    }
    /**
     * Transform Type: 
     * PS: just use power spectrum. 
     * ACF: just use autocorrelation
     * FFT: use the complex FFT terms
     * ACF_PS: Use the lot: ACF, PACF, AR and PS
     */
    public enum TransformType{PS,ACF,FFT,ACF_PS};
    TransformType transform=TransformType.ACF_PS;
    /**
    * Changes the base classifier,
    * @param c new base classifier
    */    
    public void setBaseClassifier(Classifier c){
        baseClassifierTemplate=c;
    }
    /**
     * 
     * @param k Ensemble size
     */
    public void setNumClassifiers(int k){
        numBaseClassifiers=k;
        
    }
    /**
    * 
    * @param filter one of the four transform types
    */    
    public void setTransformType(TransformType filter){
        transform=filter;
    }
    /**
     * 
     * @param s String indicating transform type
     */   
    public void setTransformType(String s){
        String str=s.toUpperCase();
        switch(str){
            case "FFT": case "DFT": case "FOURIER":
              transform=TransformType.FFT;
                break;
            case "ACF": case "AFC": case "AUTOCORRELATION":
              transform=TransformType.ACF;                
                break;
            case "PS": case "POWERSPECTRUM":
              transform=TransformType.PS;
                break;
            case "PS_ACF": case "ACF_PS": case "BOTH":
              transform=TransformType.ACF_PS;
                break;
                
        }
    }
    /**
     * 
     * @return number of classifiers in the ensemble
     */
    public int getNumClassifiers(){ 
        return numBaseClassifiers;
    }
    /**
     * Holders for the headers of each transform. 
     */    
    Instances[] testHolders;
    public RISE(){
        rand=new Random();
    }
    public RISE(int seed){
        rand=new Random();
        this.seed=seed;
        rand.setSeed(seed);
        
    }
    public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation 	result;
    result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
    result.setValue(TechnicalInformation.Field.AUTHOR, "J. Lines, S. Taylor and A. Bagnall");
    result.setValue(TechnicalInformation.Field.YEAR, "2018");
    result.setValue(TechnicalInformation.Field.TITLE, "Time series classification with HIVE-COTE: The hierarchical vote collective of transformation-based ensembles");
    result.setValue(TechnicalInformation.Field.JOURNAL, "ACM Transactions on Knowledge Discovery from Data ");
    result.setValue(TechnicalInformation.Field.VOLUME, "12");
    result.setValue(TechnicalInformation.Field.NUMBER, "5");
    result.setValue(TechnicalInformation.Field.PAGES, "NA");
    
    return result;
  }

    @Override
    public String getParameters(){
        return super.getParameters()+",numTrees,"+numBaseClassifiers+","+"MinInterval,"+minInterval;
    }
    /**
       * Returns default capabilities of the classifier. These are that the 
       * data must be numeric, with no missing and a nominal class
       * @return the capabilities of this classifier
    **/    
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        // attributes must be numeric
        // Here add in relational when ready
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }
         
    @Override
    public void buildClassifier(Instances data) throws Exception {
        // can classifier handle the data?
        getCapabilities().testWithFail(data);
//      TO DO  trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        long start=System.currentTimeMillis();
        //Option to sub sample for training        
         if(subSample){
            data=subSample(data,sampleProp,seed);
            System.out.println(" TRAIN SET SIZE NOW "+data.numInstances());
        }
        //Initialise the memory 
        int m=data.numAttributes()-1;
        startPoints =new int[numBaseClassifiers];
        endPoints =new int[numBaseClassifiers];
 
        baseClassifiers=new Classifier[numBaseClassifiers];
        testHolders=new Instances[numBaseClassifiers];
        //Select random intervals for each tree
        for(int i=0;i<numBaseClassifiers;i++){
            //Do whole series for first classifier            
            if(i==0){
                startPoints[i]=0;
                endPoints[i]=m-1;
            }
            else{
                //Random interval at least minInterval in size
                startPoints[i]=rand.nextInt(m-minInterval);
                //This avoid calling nextInt(0)
                if(startPoints[i]==m-1-minInterval) 
                    endPoints[i]=m-1;
                else{    
                    endPoints[i]=rand.nextInt(m-startPoints[i]);
                    if(endPoints[i]<minInterval)
                        endPoints[i]=minInterval;
                    endPoints[i]+=startPoints[i];
                }
            }
            //Set up train instances and save format for testing. 
            int numFeatures=endPoints[i]-startPoints[i]+1;
            String name;
            ArrayList<Attribute> atts=new ArrayList();
            for(int j=0;j<numFeatures;j++){
                    name = "F"+j;
                    atts.add(new Attribute(name));
            }
            //Get the class values as a fast vector			
            Attribute target =data.attribute(data.classIndex());
            ArrayList<String> vals=new ArrayList<>(target.numValues());
            for(int j=0;j<target.numValues();j++)
                    vals.add(target.value(j));
            atts.add(new Attribute(data.attribute(data.classIndex()).name(),vals));
            //create blank instances with the correct class value                
            Instances result = new Instances("Tree",atts,data.numInstances());
            result.setClassIndex(result.numAttributes()-1);
            for(int j=0;j<data.numInstances();j++){
                DenseInstance in=new DenseInstance(result.numAttributes());
                double[] v=data.instance(j).toDoubleArray();
                for(int k=0;k<numFeatures;k++)
                    in.setValue(k,v[startPoints[i]+k]);
            //Set interval features                
                in.setValue(result.numAttributes()-1,data.instance(j).classValue());
                result.add(in);
            }
            testHolders[i] =new Instances(result,0);       
            DenseInstance in=new DenseInstance(result.numAttributes());
            testHolders[i].add(in);
            //Perform the transform
            Instances newTrain=result;
            //THIS NEEDS TIDYING UP    
            switch(transform){
                case ACF:
                    newTrain=ACF.formChangeCombo(result);
                    break;
                case PS: 
                    newTrain=ps.process(result);
                    break;
                case ACF_PS: default:
                    newTrain=combinedPSACF(result);
                    break;
            }             
//Build Classifier: Defaults to a RandomTree, but WHY ALL THE ATTS?
            if(baseClassifierTemplate instanceof RandomTree){
                baseClassifiers[i]=new RandomTree();   
                ((RandomTree)baseClassifiers[i]).setKValue(numFeatures);
            }
            else
               baseClassifiers[i]=AbstractClassifier.makeCopy(baseClassifierTemplate);
            baseClassifiers[i].buildClassifier(newTrain);
        }
        trainResults.setBuildTime(System.currentTimeMillis()-start);
    }

    @Override
    public double[] distributionForInstance(Instance ins) throws Exception {
        double[] votes=new double[ins.numClasses()];
////Build instance
        double[] series=ins.toDoubleArray();
        for(int i=0;i<baseClassifiers.length;i++){
            int numFeatures=endPoints[i]-startPoints[i]+1;
        //extract the interval
            for(int j=0;j<numFeatures;j++){
                testHolders[i].instance(0).setValue(j, ins.value(j+startPoints[i]));
            }
//Do the transform
            Instances temp=null;
            switch(transform){
                case ACF:
                    temp=ACF.formChangeCombo(testHolders[i]);
                    break;
                case PS: 
                    temp=ps.process(testHolders[i]);
                    break;
                case ACF_PS: 
                    temp=combinedPSACF(testHolders[i]);
//Merge newTrain and newTrain2                    
                    break;
            }             
            int c=(int)baseClassifiers[i].classifyInstance(temp.instance(0));
            votes[c]++;
            
        }
        for(int i=0;i<votes.length;i++)
            votes[i]/=baseClassifiers.length;
        return votes;
    }
   private Instances combinedPSACF(Instances data)throws Exception {
        Instances combo=ACF.formChangeCombo(data);
        Instances temp2=ps.process(data);
        combo.setClassIndex(-1);
        combo.deleteAttributeAt(combo.numAttributes()-1); 
        combo=Instances.mergeInstances(combo, temp2);
        combo.setClassIndex(combo.numAttributes()-1);
        return combo;        

    }    
    
    public static void main(String[] arg) throws Exception{
        
        Instances train=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\ItalyPowerDemand\\ItalyPowerDemand_TRAIN");
        Instances test=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\ItalyPowerDemand\\ItalyPowerDemand_TEST");
        RISE rif = new RISE();

        
        rif.buildClassifier(train);
        System.out.println("build ok:");
        double a=ClassifierTools.accuracy(test, rif);
        System.out.println(" Accuracy ="+a);
/*
        //Get the class values as a fast vector			
        Attribute target =data.attribute(data.classIndex());

        FastVector vals=new FastVector(target.numValues());
        for(int j=0;j<target.numValues();j++)
                vals.addElement(target.value(j));
        atts.addElement(new Attribute(data.attribute(data.classIndex()).name(),vals));
//Does this create the actual instances?                
        Instances result = new Instances("Tree",atts,data.numInstances());
        for(int i=0;i<data.numInstances();i++){
            DenseInstance in=new DenseInstance(result.numAttributes());
            result.add(in);
        }
        result.setClassIndex(result.numAttributes()-1);
        Instances testHolder =new Instances(result,10);       
//For each tree   
        System.out.println("Train size "+result.numInstances());
        System.out.println("Test size "+testHolder.numInstances());
*/
    }
}
