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

package timeseriesweka.filters;


import experiments.SimulationExperiments;
import fileIO.OutFile;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.*;

/** 
  <!-- globalinfo-start -->
* Implementation of autocorrelation function as a Weka SimpleBatchFilter
* Series to series transform independent of class value
*  <!-- globalinfo-end -->
<!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -L
 *  set the max lag.</pre>
 <!-- options-end -->
 *
 * 
 * author: Anthony Bagnall circa 2008.
 * Reviewed and tidied up 2019
 * This should not really be a batch filter, as it is series to series, but
 * it makes the use case simpler. 
 */

public class ACF extends SimpleBatchFilter {

    private static final long serialVersionUID = 1L;
/** If the series are normalised, the calculation can be done much more efficiently*/
    private boolean normalized=false;   //if true, assum zero mean and unit variance

 /** Whatever the maxLag value, we always ignore at least the endTerms correlations 
 * since they are based on too little data and hence unreliable  */
    private  int endTerms=4;
    
/** The maximum number of ACF terms considered. It must be less than seriesLength-endTerms
 * (checked in process() */    
    public static final int DEFAULT_MAXLAG=100;
    private int maxLag=DEFAULT_MAXLAG;
/** Currently assumed constant for all series. Have to, using instances* */   
    private int seriesLength;

    public void setMaxLag(int n){ maxLag=n;}
    public void setNormalized(boolean flag){ normalized=flag;}

    /**
     * Sets up the header info for the transformed series
     * @param inputFormat
     * @return
     * @throws Exception 
     */
    @Override
    protected Instances determineOutputFormat(Instances inputFormat)  throws Exception {
    //Check capabilities for the filter. Can only handle real valued, no missing.       
        getCapabilities().testWithFail(inputFormat);
   
        seriesLength=inputFormat.numAttributes();	
        if(inputFormat.classIndex()>=0)
            seriesLength--;
//Cannot include the final endTerms correlations, since they are based on too little data and hence unreliable.
        if(maxLag>seriesLength-endTerms)
            maxLag=seriesLength-endTerms;
        if(maxLag<0)
            maxLag=inputFormat.numAttributes()-1;
        //Set up instances size and format. 
        ArrayList<Attribute> atts=new ArrayList<>();
        String name;
        for(int i=1;i<=maxLag;i++){
            name = "ACF_"+i;
            atts.add(new Attribute(name));
        }
        if(inputFormat.classIndex()>=0){
            //Get the class values as an ArrayList
            Attribute target =inputFormat.attribute(inputFormat.classIndex());

            ArrayList<String> vals=new ArrayList<>(target.numValues());
            for(int i=0;i<target.numValues();i++)
                    vals.add(target.value(i)+"");
            atts.add(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(),vals));
        }	
        Instances result = new Instances("ACF"+inputFormat.relationName(),atts,inputFormat.numInstances());
        if(inputFormat.classIndex()>=0){
            result.setClassIndex(result.numAttributes()-1);
        }
        return result;
    }
    @Override
    public String globalInfo() {    return " Autocorrelation function";    }
/**
   * Parses a given list of options. <p/>
   * 
   <!-- options-start -->
   * Valid options are: <p/>
   * <pre> -L &lt;num&gt;
   *  max lag for the ACF function</pre>
   <!-- options-end -->
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
    @Override
    public void setOptions(String[] options) throws Exception {
        String maxLagString=Utils.getOption('L', options);
        if (maxLagString.length() != 0)
            this.maxLag = Integer.parseInt(maxLagString);
        else
            this.maxLag = DEFAULT_MAXLAG;
    
    }
    
    /**
     * ACF can only operate on real valued attributes with no missing values
     * @return Capabilities object
     */    
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        // attributes must be numeric
        // Here add in relational when ready
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
    //    result.enable(Capabilities.Capability.MISSING_VALUES);

        // class
        result.enableAllClasses();
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
        result.enable(Capabilities.Capability.NO_CLASS);

        return result;
    }

    /**
     * NOTE THIS SHOULD BE PROTECTED. We have hacked the Weka base class. This 
     * needs revising.
     * @param inst Instances series to convert
     * @return Instances where series to series ACF has been conducted for maxLag terms
     * @throws Exception 
     */
    @Override
    public Instances process(Instances inst) throws Exception {
        Instances output=determineOutputFormat(inst);
        //For each data, first extract the relevant series
        seriesLength=inst.numAttributes();
        int acfLength=output.numAttributes();
        if(inst.classIndex()>=0){
            seriesLength--;
            acfLength--;
        }
        for(int i=0;i<inst.numInstances();i++){
        //1. Get series: 
            double[] d=inst.instance(i).toDoubleArray();
            //2. Remove target class
            double[] temp;
            int c=inst.classIndex();
            if(c>=0){
                temp=new double[d.length-1];
                int count=0;
                for(int k=0;k<d.length;k++){
                    if(k!=c)
                        temp[count++]=d[k];
                }
                d=temp;
            }
            double[] autoCorr=fitAutoCorrelations(d);
//Set the attributes
            Instance newInst=null;
            if(inst.classIndex()>=0)
                newInst=new DenseInstance(acfLength+1);
            else
                newInst=new DenseInstance(acfLength);

            for(int j=0;j<acfLength;j++){
//CHECK THIS VS ALTERNATIVE ALGORITHMS. ASurely it cannot be <1 or >1??                
                if(autoCorr[j]<-1.0 || autoCorr[j]>1 || Double.isNaN(autoCorr[j])|| Double.isInfinite(autoCorr[j]))
                    newInst.setValue(j,0);
                else
                    newInst.setValue(j,autoCorr[j]);
            }
            if(inst.classIndex()>=0)
                newInst.setValue(output.classIndex(), inst.instance(i).classValue());
            output.add(newInst);
        }
        return output;
    }
/**
 * Note it is possible to do this with FFT in O(nlogn) BUT requires 2^n attributes
 * @param data
 * @return 
 */    
    public double[] fitAutoCorrelations(double[] data) {
        double[] a = new double[maxLag];
        if(!normalized){
            for(int i=1;i<=maxLag;i++){
                double s1,s2,ss1,ss2,v1,v2;
                a[i-1]=0;
                s1=s2=ss1=ss2=0;
                for(int j=0;j<data.length-i;j++){
                    s1+=data[j];
                    ss1+=data[j]*data[j];
                    s2+=data[j+i];
                    ss2+=data[j+i]*data[j+i];
                }
                s1/=data.length-i;
                s2/=data.length-i;
                for(int j=0;j<data.length-i;j++)
                    a[i-1]+=(data[j]-s1)*(data[j+i]-s2);
                a[i-1]/=(data.length-i);
                v1=ss1/(data.length-i)-s1*s1;
                v2=ss2/(data.length-i)-s2*s2;
                if(v1==0 && v2==0)//Both zero variance, both must be 100% corr
                    a[i-1]=1;
                else if(v1==0 || (v2==0))//One zero variance the other not
                    a[i-1]=0;
                else
                    a[i-1]/=Math.sqrt(v1)*Math.sqrt(v2);
            }
        }
        else{
            for(int i=1;i<=maxLag;i++){
                a[i-1]=0;
                for(int j=0;j<data.length-i;j++)
                    a[i-1]+=data[j]*data[j+i];
                a[i-1]/=data.length;
            }
        }
        return a;
    }
/**
 * Static variant, with no normalisation speed up.
 * @param data
 * @param mLag
 * @return first mLag autocorrelations
 */    
    public static double[] fitAutoCorrelations(double[] data, int mLag){
        return fitAutoCorrelations(data,mLag,false);
    }
    
    public static double[] fitAutoCorrelations(double[] data, int mLag, boolean normalised){
        double[] a = new double[mLag];

        if(!normalised){
            double s1,s2,ss1,ss2,v1,v2;
            for(int i=1;i<=mLag;i++){
                a[i-1]=0;
                s1=s2=ss1=ss2=0;
                for(int j=0;j<data.length-i;j++){
                    s1+=data[j];
                    ss1+=data[j]*data[j];
                    s2+=data[j+i];
                    ss2+=data[j+i]*data[j+i];
                }
                s1/=data.length-i;
                s2/=data.length-i;
                for(int j=0;j<data.length-i;j++)
                    a[i-1]+=(data[j]-s1)*(data[j+i]-s2);
                a[i-1]/=(data.length-i);
                v1=ss1/(data.length-i)-s1*s1;
                v2=ss2/(data.length-i)-s2*s2;
                if(v1!=0 && v2!=0)
                    a[i-1]/=Math.sqrt(v1)*Math.sqrt(v2);
            }
        }
        else{
            for(int i=1;i<=mLag;i++){
                a[i-1]=0;
                for(int j=0;j<data.length-i;j++)
                    a[i-1]+=data[j]*data[j+i];
                a[i-1]/=data.length;
            }
        }
        
        return a;
    }


    public String getRevision() {
        return "Revision 2: 2019";
    }


    
    
/**
 * Below are Instances level functions to reshape the whole data set
 * based on characteristics of all series.
 */    
/**
 * Pre: An Instances of ACF transformed data. 
 * Finds the indexes of the last significant feature for all instances 
 * @param inst
 * @return array of integer indexes
 */    
/** 
* These are data set level options
*/
    public void setGlobalSigThresh(boolean flag){ useGlobalSigThreshold=flag;}
    int globalSignificantLag=maxLag;
    double globalSigThreshold;
    boolean useGlobalSigThreshold=true;
    double[] sigThreshold;
    int[] cutOffs;
    boolean globalTruncate=true;
    double alpha=0.1;   // Significant threshold for the truncation
    
    
    public int truncate(Instances d, boolean global){
        globalTruncate=global;
        return truncate(d);

    }
    /**
     * Firstly, this method finds the first insignificant ACF term in every series
     * It then does does one of two things
     *  if globalTruncate is true, it finds the max cut off point, and truncates all to thisd
     *  if not, it zeros all values after the truncation point. 
     * @param d
     * @return largest cut off point
     */
    public int truncate(Instances d){
    //Truncate 1: find the first insignificant term for each series, then find the highest, then remove all after this    
        int largestPos=0;
        int[] c=findAllCutOffs(d);
        if(globalTruncate){
            for(int i=1;i<c.length;i++){
                if(c[largestPos]<c[i])
                    largestPos=i;
            }
//This is to stop zero attributes!
            if(largestPos<d.numAttributes()-2)
                largestPos++;
            truncate(d,largestPos);
        }
        else{
            for(int i=0;i<d.numInstances();i++){
                zeroInstance(d.instance(i),c[i]);
            }
        }
        return largestPos;
    }
/**
 * use
 * @param inst
 * @return 
 */    
    
    private int[] findAllCutOffs(Instances inst){

        globalSigThreshold=2/Math.sqrt(seriesLength);
        sigThreshold=new double[inst.numAttributes()-1];
        cutOffs=new int[inst.numInstances()];
        for(int i=0;i<cutOffs.length;i++)
            cutOffs[i]=findSingleCutOff(inst.instance(i));
        return cutOffs;
    }
/**
    PRE: An instance of ACF data. Performs a test of significance on the 
    * ACF terms until it finds the first insignificant one.
    * Will not work if the class variable is not 
    the last. 
    * @param inst
 * @return 
 */    
    private int findSingleCutOff(Instance inst){
/** Finds the threshold of the first non significant ACF term for all the series.
*/            
        double[] r=inst.toDoubleArray();
        int count=0;
        if(useGlobalSigThreshold){
            for(int i=0;i<inst.numAttributes();i++){
                if(i!=inst.classIndex()){
                    sigThreshold[count]=globalSigThreshold;
                    count++;
                }
            }
        }
        else{   ///DO NOT USE, I'm not sure of the logic of this, need to look up the paper
            sigThreshold[0]=r[0]*r[0];
            count=1;
            for(int i=1;i<inst.numAttributes();i++){
                if(i!=inst.classIndex()){
                sigThreshold[count]=sigThreshold[count-1]+r[i]*r[i]; 
                count++;
                }
            }
            for(int i=0;i<sigThreshold.length;i++){
                sigThreshold[i]=(1+sigThreshold[i])/seriesLength;
                sigThreshold[i]=2/Math.sqrt(sigThreshold[i]);
            }
        }
        for(int i=0;i<sigThreshold.length;i++)
            if(Math.abs(r[i])<sigThreshold[i])
                return i;
        return sigThreshold.length-1;
    }
/**
 * Truncates all cases to having n attributes, i.e. removes from numAtts()-n to numAtts()-1  
 * @param d
 * @param n 
 */    
    public void truncate(Instances d, int n){
        int att=n;
        while(att<d.numAttributes()){
            if(att==d.classIndex())
                att++;
            else
                d.deleteAttributeAt(att);
        }
    }
/**
 * Sets all values from p to end to zero
 * @param ins
 * @param p 
 */    
    private void zeroInstance(Instance ins, int p){
        for(int i=p;i<ins.numAttributes();i++){
            if(i!=ins.classIndex())
                ins.setValue(i, 0);
        }
    }
/**
/**Debug code to test ACF generation: 
 */
    public static void testTransform(){
//    Test File ACF: Four AR(1) series, first two \phi_0=0.5, seconde two \phi_0=-0.5
        Instances test=ClassifierTools.loadData("C:\\Research\\Data\\TestData\\ACFTest");
        DecimalFormat df=new DecimalFormat("##.####");
        ACF acf=new ACF();
        acf.setMaxLag(test.numAttributes()-10);
        try{
            Instances t2=acf.process(test);
            System.out.println(" Number of attributes ="+t2.numAttributes());
            Instance ins=t2.instance(0);
            for(int i=0;i<ins.numAttributes()&&i<10;i++)
                    System.out.print(" "+df.format(ins.value(i)));
            OutFile of=new OutFile("C:\\Research\\Data\\TestData\\ACTTestOutput.csv");
            of.writeString(t2.toString());
        }catch(Exception e){
            System.out.println(" Exception in ACF harness="+e);
            e.printStackTrace();
        }
    } 
    public static void testTrunctate(){
        Instances test=ClassifierTools.loadData("C:\\Research\\Data\\TestData\\ACFTest");
        DecimalFormat df=new DecimalFormat("##.####");
        ACF acf=new ACF();
        int[] cases={20,20};
        int seriesLength=200;
        acf.setMaxLag(test.numAttributes()-10);
        try{
            acf.setMaxLag(seriesLength-10);
            Instances all=SimulationExperiments.simulateData("AR",1);
            System.out.println(" Number of attributes All ="+all.numAttributes());
            Instances t2=acf.process(all);
            System.out.println(" Number of attributes ="+t2.numAttributes());
            acf.truncate(t2);
            System.out.println(" Number of attributes ="+t2.numAttributes());
            acf.useGlobalSigThreshold=true;
            t2=acf.process(all);
            acf.truncate(t2);
            System.out.println(" Number of attributes ="+t2.numAttributes());
        }catch(Exception e){
            System.out.println(" Exception in ACF harness="+e);
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        String problemPath = "E:/TSCProblems/";
        String resultsPath="E:/Temp/";
        String datasetName="ItalyPowerDemand";
        Instances train =ClassifierTools.loadData("E:/TSCProblems/"+datasetName+"/"+datasetName+"_TRAIN");
        ACF acf= new ACF();
        try {
            Instances trans=acf.process(train);
            OutFile out = new OutFile(resultsPath+datasetName+"ACF_JAVA.csv");
            out.writeLine(datasetName);
            for(Instance ins: trans){
                double[] d=ins.toDoubleArray();
                for(int j=0;j<d.length;j++){
                    if(j!=trans.classIndex())
                        out.writeString(d[j]+",");
                }
                out.writeString("\n");
            }
        } catch (Exception ex) {
            Logger.getLogger(ACF.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
	