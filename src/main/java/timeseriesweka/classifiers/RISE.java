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

/**
 <!-- globalinfo-start -->
 * Random Interval Spectral Ensemble
 *
 * This implementation extends the original to include:
 * down sampling
 * stabilisation (constraining interval length to within some distance of previous length)
 * check pointing
 * contracting
 *
 * Overview: Input n series length m
 * for each tree
 *      sample interval of random size
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
 <!-- options-end -->
 * @author Tony Bagnall
 * @date Some time in 2017
 **/


public class RISE extends AbstractClassifierWithTrainingData implements SaveParameterInfo, SubSampleTrain{
    long buildTime;
    Classifier[] baseClassifiers;
    Classifier baseClassifierTemplate=new RandomTree();
    int numBaseClassifiers=500;
//INTERVAL BOUNDS ARE INCLUSIVE    
    int[] startPoints;
    int[] endPoints;
    public static int MIN_INTERVAL=16;
    public static int MIN_BITS=3;
    Random rand;
    PowerSpectrum ps=new PowerSpectrum();
    private boolean subSample=false;
    private double sampleProp=1;
    private int sampleSeed=0;
    

    public void subSampleTrain(double prop, int s){
        subSample=true;
        sampleProp=prop;
        sampleSeed=s;
    }
    public void setBaseClassifier(Classifier c){
        baseClassifierTemplate=c;
    }

    public void setNumTrees(int i) {
        numBaseClassifiers=i;
    }
    
    public enum TransformType{PS,ACF,FFT,ACF_PS};
    TransformType f=TransformType.ACF_PS;
    public void setTransformType(TransformType fil){
        f=fil;
    }
    
    public void setTransformType(String s){
        String str=s.toUpperCase();
        switch(str){
            case "FFT": case "DFT": case "FOURIER":
              f=TransformType.FFT;
                break;
            case "ACF": case "AFC": case "AUTOCORRELATION":
              f=TransformType.ACF;                
                break;
            case "PS": case "POWERSPECTRUM":
              f=TransformType.PS;
                break;
            case "PS_ACF": case "ACF_PS": case "BOTH":
              f=TransformType.ACF_PS;
                break;
                
        }
    }
    public void setNosBaseClassifiers(int n){
        numBaseClassifiers=n;
    }
    public int getNosBaseClassifiers(){ 
        return numBaseClassifiers;
    }
    
    Instances[] testHolders;
    public RISE(){
        rand=new Random();
    }
    public RISE(int seed){
        rand=new Random();
        rand.setSeed(seed);
    }
    public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation 	result;
    result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
    result.setValue(TechnicalInformation.Field.AUTHOR, "A. Bagnall");
    result.setValue(TechnicalInformation.Field.YEAR, "2016");
    result.setValue(TechnicalInformation.Field.TITLE, "Not published");
    result.setValue(TechnicalInformation.Field.JOURNAL, "NA");
    result.setValue(TechnicalInformation.Field.VOLUME, "NA");
    result.setValue(TechnicalInformation.Field.PAGES, "NA");
    
    return result;
  }

    @Override
    public String getParameters(){
        return super.getParameters()+",numTrees,"+numBaseClassifiers+","+"MinInterval"+MIN_INTERVAL;
    }
         
    @Override
    public void buildClassifier(Instances data) throws Exception {
        long start=System.currentTimeMillis();

//Estimate Train CV, store CV     
         if(subSample){
            data=subSample(data,sampleProp,sampleSeed);
            System.out.println(" TRAIN SET SIZE NOW "+data.numInstances());
        }
        
//Determine the number of baseClassifiers, max the number of attributes. 
//        if(data.numAttributes()-1<numBaseClassifiers)
//           numBaseClassifiers=data.numAttributes()-1;
//Set series length
        int m=data.numAttributes()-1;
        startPoints =new int[numBaseClassifiers];
        endPoints =new int[numBaseClassifiers];
        baseClassifiers=new Classifier[numBaseClassifiers];
        testHolders=new Instances[numBaseClassifiers];
        //1. Select random intervals for each tree
        for(int i=0;i<numBaseClassifiers;i++){
            if(i==0){//Do whole series for first classifier
                startPoints[i]=0;
                endPoints[i]=m-1;
            }
            else{
                startPoints[i]=rand.nextInt(m-MIN_INTERVAL);
                if(startPoints[i]==m-1-MIN_INTERVAL) 
//Interval at the end, need to avoid calling nextInt with argument 0
                    endPoints[i]=m-1;
                else{    
                    endPoints[i]=rand.nextInt(m-startPoints[i]);
                    if(endPoints[i]<MIN_INTERVAL)
                        endPoints[i]=MIN_INTERVAL;
                    endPoints[i]+=startPoints[i];
                }
            }

//            System.out.println("START = "+startPoints[i]+" END ="+endPoints[i]);
//Set up train instances and save format for testing. 
            int numFeatures=endPoints[i]-startPoints[i]+1;
            String name;
            FastVector atts=new FastVector();
            for(int j=0;j<numFeatures;j++){
                    name = "F"+j;
                    atts.addElement(new Attribute(name));
            }
            //Get the class values as a fast vector			
            Attribute target =data.attribute(data.classIndex());
            FastVector vals=new FastVector(target.numValues());
            for(int j=0;j<target.numValues();j++)
                    vals.addElement(target.value(j));
            atts.addElement(new Attribute(data.attribute(data.classIndex()).name(),vals));
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
            
            switch(f){
                case ACF:
                    newTrain=ACF.formChangeCombo(result);
                    break;
                case PS: 
                    newTrain=ps.process(result);
                    break;
                case ACF_PS: 
                    newTrain=combinedPSACF(result);
//Merge newTrain and newTrain2                    
                    break;
                    
            }             
//Build Classifier: Defaults to a RandomTree, but want to tre
            if(baseClassifierTemplate instanceof RandomTree){
                baseClassifiers[i]=new RandomTree();   
                ((RandomTree)baseClassifiers[i]).setKValue(numFeatures);
            }
            else
               baseClassifiers[i]=AbstractClassifier.makeCopy(baseClassifierTemplate);
            baseClassifiers[i].buildClassifier(newTrain);
        }
        trainResults.setBuildTime(System.nanoTime()-start);
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
            switch(f){
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
    @Override
    public double classifyInstance(Instance ins) throws Exception {
        int[] votes=new int[ins.numClasses()];
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
            switch(f){
                case ACF:
                    temp=ACF.formChangeCombo(testHolders[i]);
                    break;
                case PS: 
                    temp=ps.process(testHolders[i]);
                    break;
                case ACF_PS: 
                    temp=combinedPSACF(testHolders[i]);//Merge newTrain and newTrain2                    
                    break;
            }             
            int c=(int)baseClassifiers[i].classifyInstance(temp.instance(0));
            votes[c]++;
        }
//Return majority vote            
       int maxVote=0;
       for(int i=1;i<votes.length;i++)
           if(votes[i]>votes[maxVote])
               maxVote=i;
       return maxVote;
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
    public static void intervalGenerationTest(){
        int m=500;
        int numTrees=500;
        int[] startPoints =new int[numTrees];
        int[] endPoints =new int[numTrees];
        Random rand=new Random();
        for(int i=0;i<numTrees;i++){
            if(i==0){//Do whole series
                startPoints[i]=0;
                endPoints[i]=m-1;
            }
            else{
                startPoints[i]=rand.nextInt(m-MIN_INTERVAL);
                if(startPoints[i]==m-1-MIN_INTERVAL) 
//Interval at the end, need to avoid calling nextInt with argument 0
                    endPoints[i]=m-1;
                else{    
                    endPoints[i]=rand.nextInt(m-startPoints[i]);
                    if(endPoints[i]<MIN_INTERVAL)
                        endPoints[i]=MIN_INTERVAL;
                    endPoints[i]+=startPoints[i];
                }
            }
            System.out.println("START = "+startPoints[i]+" END ="+endPoints[i]+ " LENGTH ="+(endPoints[i]-startPoints[i]));
        }
    }

    public static int[] setIntervalLengths(int max){
//Find max power of 2 less than max        
        int test=(int)(Math.log(Integer.highestOneBit(max))/Math.log(2));
        int[] lengths;
        int l=2;
        if(test>MIN_BITS){
            lengths=new int[test-MIN_BITS+1];
            for(int i=1;i<MIN_BITS;i++)
                l*=2;
            lengths[0]=l;
            for(int i=1;i<lengths.length;i++)
                lengths[i]=lengths[i-1]*2;
        }
        else{
            lengths=new int[1];
            for(int i=1;i<test;i++)
                l*=2;
            lengths[0]=l;
        }
            
        return lengths;
    } 
    
     
    
    public static void main(String[] arg) throws Exception{
//        intervalGenerationTest();
//        System.exit(0);
        
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
