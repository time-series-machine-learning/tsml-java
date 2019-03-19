package timeseriesweka.classifiers;
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
import timeseriesweka.filters.ARMA;
import timeseriesweka.filters.PACF;
import weka.core.Capabilities;
import weka.core.Randomizable;
import weka.core.Utils;
import weka.filters.Filter;

/**
 * Development code for RISE
 * 1. set number of trees to max(500,m)
 * 2. Set the first tree to the full interval
 * 2. Randomly select the interval length and start point for each other tree *
 * 3. Find the PS, ACF, PACF and AR features
 * 3. Build each tree.

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
    private Classifier baseClassifierTemplate=new RandomTree();
    /** Ensemble base classifiers */    
    private Classifier[] baseClassifiers;
    /** Ensemble size */    
    private static int DEFAULT_NUM_CLASSIFIERS=500;
    private int numBaseClassifiers=DEFAULT_NUM_CLASSIFIERS;
    /** Random Intervals for the transform. INTERVAL BOUNDS ARE INCLUSIVE  */  
    private int[] startPoints;
    private int[] endPoints;
    /** Minimum size of all intervals */    
    private static int DEFAULT_MIN_INTERVAL=16;
    private int minInterval=DEFAULT_MIN_INTERVAL;
    
    /**Can seed for reproducibility */
    private Random rand;
    private int seed=0;
    private boolean setSeed=false;
    Filter[] filters;
    /** Power Spectrum transformer, probably dont need to store this here  */   
//    private PowerSpectrum ps=new PowerSpectrum();
    
    /** If we are estimating the CV, it is possible to sample fewer elements. **/
    //Really should try bagging this!    
    private boolean subSample=false;
    private double sampleProp=1;
    public RISE(){
        filters=new Filter[4];
        filters[0]=new ACF();
        filters[1]=new PACF();
        filters[2]=new ARMA();
        filters[3]=new PowerSpectrum();
    }
    public RISE(int s){
        this();
        seed=s;
    }
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


    public void setTransforms(String ... trans){
        filters=new Filter[trans.length];
        int count=0;
        for(String s:trans){
            switch(s){
                case "ACF": case "Autocorrelation":
                    filters[count]= new ACF();
                    break;
                case "PACF": case "PartialAutocorrelation":
                    filters[count]= new PACF();
                    break;
                case "AR": case "AutoRegressive":
                    filters[count]= new ARMA();
                    break;
                case "PS": case "PowerSpectrum":
                    filters[count]= new PowerSpectrum();
                    break;
                default:
                    System.out.println("Unknown tranform "+s);
                    continue;
            }
            count++;
        }
        if(count<filters.length){
            Filter[] temp=new Filter[count];
            for(int i=0;i<count;i++)
                temp[i]=filters[i];
            filters=temp;
        }
    }
    
    /**
     * Transform Type: 
     * PS: just use power spectrum. 
     * ACF: just use autocorrelation
     * FFT: use the complex FFT terms
     * ACF_PS: Use the lot: ACF, PACF, AR and PS
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
     * @return number of classifiers in the ensemble
     */
    public int getNumClassifiers(){ 
        return numBaseClassifiers;
    }
    /**
     * Holders for the headers of each transform. 
     */    
    Instances[] testHolders;
    public void setSeed(int s){
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
  /**
   * Parses a given list of options to set the parameters of the classifier.
   * We use this for the tuning mechanism, setting parameters through setOptions 
   <!-- options-start -->
   * Valid options are: <p/>
   * <pre> -K
   * Number of base classifiers.
   * </pre>
   * <pre> -I
   * min Interval, integer, should be in range 3 to m-MINa check in build classifier is made to see if if.
   * </pre>
   * <pre> -T
        transforms, a space separated list.
   * </pre>
   * 
   <!-- options-end -->
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
    @Override
    public void setOptions(String[] options) throws Exception{
        String numCls=Utils.getOption('K', options);
        if (numCls.length() != 0)
            numBaseClassifiers = Integer.parseInt(numCls);
        else
            numBaseClassifiers = DEFAULT_NUM_CLASSIFIERS;
    /** Minimum sizer of all intervals */    
        String minInt=Utils.getOption('I', options);
        if (minInt.length() != 0)
            minInterval=Integer.parseInt(minInt);
    /** Transforms to use */    
        String trans=Utils.getOption('T', options);
        String[] t= trans.split(" ");
//NEED TO CHECK THIS WORKS        
        setTransforms(t);
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
        int m=data.numAttributes()-1;
        startPoints =new int[numBaseClassifiers];
        endPoints =new int[numBaseClassifiers];
//      TO DO  trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        long start=System.currentTimeMillis();
        //Option to sub sample for training        
         if(subSample){
            data=subSample(data,sampleProp,seed);
            System.out.println(" TRAIN SET SIZE NOW "+data.numInstances());
        }
         
        //Initialise the memory 
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
            //Set up train instances prior to trainsform.
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
            for(Filter f:filters){
                Instances ins=Filter.useFilter(result, f);
            }
            
            
/*            
            //THIS NEEDS TIDYING UP    
            switch(transform){
                case ACF:
                    ACF acf = new ACF();
                    newTrain=formChangeCombo(result);
                    break;
                case PS: 
                    newTrain=ps.process(result);
                    break;
                case ACF_PS: default:
                    newTrain=combinedPSACF(result);
                    break;
            }             
*/            
//Build Classifier: Defaults to a RandomTree, but WHY ALL THE ATTS?
            if(baseClassifierTemplate instanceof RandomTree){
                baseClassifiers[i]=new RandomTree();   
                ((RandomTree)baseClassifiers[i]).setKValue(numFeatures);
            }
            else
               baseClassifiers[i]=AbstractClassifier.makeCopy(baseClassifierTemplate);
            //if(baseClassifiers[i] instanceof Randomisable)
            if(baseClassifiers[i] instanceof Randomizable)
                ((Randomizable)baseClassifiers[i]).setSeed(i*seed);
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
/*            
            switch(transform){
                case ACF:
                    temp=formChangeCombo(testHolders[i]);
                    break;
                case PS: 
                    temp=ps.process(testHolders[i]);
                    break;
                case ACF_PS: 
                    temp=combinedPSACF(testHolders[i]);
//Merge newTrain and newTrain2                    
                    break;
            }            
*/            
            int c=(int)baseClassifiers[i].classifyInstance(temp.instance(0));
            votes[c]++;
            
        }
        for(int i=0;i<votes.length;i++)
            votes[i]/=baseClassifiers.length;
        return votes;
    }
   private Instances combinedPSACF(Instances data)throws Exception {
        Instances combo=formChangeCombo(data);
//        Instances temp2=ps.process(data);
        combo.setClassIndex(-1);
        combo.deleteAttributeAt(combo.numAttributes()-1); 
  //      combo=Instances.mergeInstances(combo, temp2);
        combo.setClassIndex(combo.numAttributes()-1);
        return combo;        

    }
   
    public Instances formChangeCombo(Instances d){
        int maxLag=(d.numAttributes()-1)/4;
        if(maxLag>ACF.DEFAULT_MAXLAG)
            maxLag=ACF.DEFAULT_MAXLAG;

        try{
            //1. ACF
            ACF acf=new ACF();
            acf.setMaxLag(maxLag);
            acf.setNormalized(false);
            Instances acfData=acf.process(d);
            //2. ARMA 
            ARMA arma=new ARMA();                        
            arma.setMaxLag(maxLag);
            arma.setUseAIC(false);
            Instances arData=arma.process(d);
            //3. PACF Full. Surely ARMA finds PACF too?!?
            PACF pacf=new PACF();
            pacf.setMaxLag(maxLag);
            Instances pacfData=pacf.process(d);
            
            //4. Merge them all together
            Instances combo=new Instances(acfData);
            combo.setClassIndex(-1);
            combo.deleteAttributeAt(combo.numAttributes()-1); 
            combo=Instances.mergeInstances(combo, pacfData);
            combo.deleteAttributeAt(combo.numAttributes()-1); 
            combo=Instances.mergeInstances(combo, arData);
            combo.setClassIndex(combo.numAttributes()-1);
            return combo;

       }catch(Exception e){
//FIX THIS           
            System.out.println(" Exception in Combo="+e+" max lag ="+maxLag);
            e.printStackTrace();
            System.exit(1);
       }
       return null;
    }
    
   
    
    public static void main(String[] arg) throws Exception{
        
        
        Instances train=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\ItalyPowerDemand\\ItalyPowerDemand_TRAIN");
        Instances test=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\ItalyPowerDemand\\ItalyPowerDemand_TEST");
        RISE rif = new RISE();
        rif.setTransforms("ACF","AR","AFC");
        for(Filter f: rif.filters)
            System.out.println(f);
        String[] temp={"PS","Autocorellation","BOB","PACF"};
        rif.setTransforms(temp);
        for(Filter f: rif.filters)
            System.out.println(f);
        System.exit(0);
        
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
