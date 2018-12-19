package timeseriesweka.filters;


//import fileIO.OutFile;
import java.text.DecimalFormat;
import java.util.logging.Level;
import java.util.logging.Logger;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.bayes.NaiveBayes;
//import simulators.SimulateAR;
//import weka.classifiers.evaluation.ClassifierTools;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.*;

/*
 *      * author: Anthony Bagnall

 */

public class ACF extends SimpleBatchFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
        private boolean normalized=false;   //Assumes zero mean and unit variance
        
	int endTerms=4;
	int maxLag=300;
        int seriesLength;
	int lag=maxLag;
        int globalSignificantLag=maxLag;
        double globalSigThreshold;
        boolean useGlobalSigThreshold=true;
        double[] sigThreshold;
        int[] cutOffs;
        boolean globalTruncate=true;
        double alpha=0.1;   // Significant threshold for the4 truncation
	public void setMaxLag(int n){ maxLag=n;}
        public void setNormalized(boolean flag){ normalized=flag;}
        public void setGlobalSigThresh(boolean flag){ useGlobalSigThreshold=flag;}
	
	protected Instances determineOutputFormat(Instances inputFormat)
                throws Exception {

                seriesLength=inputFormat.numAttributes();	
                if(inputFormat.classIndex()>=0)
                    seriesLength--;
                //Check all attributes are real valued, otherwise throw exception
                for(int i=0;i<inputFormat.numAttributes();i++)
                        if(inputFormat.classIndex()!=i)
                                if(!inputFormat.attribute(i).isNumeric())
                                        throw new Exception("Non numeric attribute not allowed in ACF");
        //Cannot include the final endTerms correlations, since they are based on too little data and hence unreliable.
                if(maxLag>inputFormat.numAttributes()-endTerms)
                    maxLag=inputFormat.numAttributes()-endTerms;
                if(maxLag<0)
                    maxLag=inputFormat.numAttributes()-1;
                //Set up instances size and format. 
                FastVector atts=new FastVector();
                String name;
                for(int i=0;i<maxLag;i++){
                        name = "ACF_"+i;
                        atts.addElement(new Attribute(name));
                }
                if(inputFormat.classIndex()>=0){	//Classification set, set class 
                        //Get the class values as a fast vector			
                        Attribute target =inputFormat.attribute(inputFormat.classIndex());

                        FastVector vals=new FastVector(target.numValues());
                        for(int i=0;i<target.numValues();i++)
                                vals.addElement(target.value(i));
                        atts.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(),vals));
                }	
                Instances result = new Instances("ACF"+inputFormat.relationName(),atts,inputFormat.numInstances());
                if(inputFormat.classIndex()>=0){
                        result.setClassIndex(result.numAttributes()-1);
                }

                return result;
        }
        @Override
        public String globalInfo() {

        return null;
        }


        @Override
        public Instances process(Instances inst) throws Exception {
                Instances output=determineOutputFormat(inst);
                //For each data, first extract the relevan
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
                                        if(k!=c){
                                                temp[count]=d[k];
                                                count++;

                                        }
                                }
                                d=temp;
                        }
                        double[] autoCorr=fitAutoCorrelations(d);


        //Extract out the terms and set the attributes
                        Instance newInst=null;
                        if(inst.classIndex()>=0)
                                newInst=new DenseInstance(acfLength+1);
                        else
                                newInst=new DenseInstance(acfLength);

                        for(int j=0;j<acfLength;j++){
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
        public double[] fitAutoCorrelations(double[] data)
        {
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
        public static double[] fitAutoCorrelations(double[] data, int mLag)
        {
                double[] a = new double[mLag];

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
                            a[i-1]/=Math.sqrt(v1)*Math.sqrt(v2);
                }
                return a;
        }


        public String getRevision() {
                // TODO Auto-generated method stub
                return null;
        }
        public int truncate(Instances d, boolean global){
            globalTruncate=global;
            return truncate(d);
            
        }

        /** Firstly, this method finds the first insignificant ACF term in every series
                * If then does does one of two things
                * if globalTruncate is true, it finds the max position of  
        **/
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
        public void truncate(Instances d, int n){
                int att=n;
                while(att<d.numAttributes()){
                        if(att==d.classIndex())
                                att++;
                        else
                                d.deleteAttributeAt(att);
                }
        }
        private void zeroInstance(Instance ins, int p){
            for(int i=p;i<ins.numAttributes();i++){
                if(i!=ins.classIndex())
                    ins.setValue(i, 0);
            }
        }

        private int[] findAllCutOffs(Instances inst){
            
            globalSigThreshold=2/Math.sqrt(seriesLength);
            sigThreshold=new double[inst.numAttributes()-1];
            cutOffs=new int[inst.numInstances()];
            for(int i=0;i<cutOffs.length;i++)
                cutOffs[i]=findSingleCutOff(inst.instance(i));
            return cutOffs;
        }
/* Assumes you pass an ACF data set! Will not work if the class variable is not the last. Two fiddly to do at the moment**/
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
   public static Instances formChangeCombo(Instances d){
            int maxLag=(d.numAttributes()-1)/4;
            if(maxLag>100)
                maxLag=100;
            if(maxLag<10)
                maxLag=(d.numAttributes()-1);
                
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
          //3. PACF Full
                PACF pacf=new PACF();
                pacf.setMaxLag(maxLag);
                Instances pacfData=pacf.process(d);
                Instances combo=new Instances(acfData);
                combo.setClassIndex(-1);
                combo.deleteAttributeAt(combo.numAttributes()-1); 
                combo=Instances.mergeInstances(combo, pacfData);
                combo.deleteAttributeAt(combo.numAttributes()-1); 
                combo=Instances.mergeInstances(combo, arData);
                combo.setClassIndex(combo.numAttributes()-1);
                return combo;

           }catch(Exception e){
			System.out.println(" Exception in Combo="+e+" max lag ="+maxLag);
			e.printStackTrace();
                        System.exit(0);
           }
           return null;
    }
    
 
        /*
    public static void testTransform(){
    /**Debug code to test ACF generation: 
	Test File ACF: Four AR(1) series, first two \phi_0=0.5, seconde two \phi_0=-0.5
 	
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
                    Instances all=SimulateAR.generateARDataSet(20,20,seriesLength,cases,true);
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
*/        
        
	public static void main(String[] args){
//		testTransform();
//                testTrunctate();
                Instances train =ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\ElectricDevices\\ElectricDevices_TRAIN");
                Instances test =ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\ElectricDevices\\ElectricDevices_TEST");
                ACF acf= new ACF();
                Instances f;
            try {
                f = acf.process(train);
//                System.out.println(f.toString());
                NaiveBayes nb=new NaiveBayes();    
//                HESCA we= new HESCA();
                f=InstanceTools.subSample(f, f.numInstances()/100, 0);
                nb.buildClassifier(f);
            } catch (Exception ex) {
                Logger.getLogger(ACF.class.getName()).log(Level.SEVERE, null, ex);
            }
                
                
                
        }


	
	
}
/*			
			new double[acfLength];
	//2. Find mean and variance
		double mean=d[0];
		double var=d[0]*d[0];
		for(int j=1;j<seriesLength;j++){
			mean+=d[j];
			var+=d[j]*d[j];
		}
		mean/=seriesLength;
		var=(mean*mean-var*seriesLength)/(seriesLength-1);
	//Work out the auto-c for lags 1 to length	
		for(int lag=1;lag<maxLag;lag++)
		{
			autoCorr[lag-1]=0;
			for(int j=0;j<acfLength;j++)
				autoCorr[lag-1]+=(d[j]-mean)*(d[j+lag]-mean);
			autoCorr[lag-1]/=(seriesLength-lag)*var;
		}
*/	