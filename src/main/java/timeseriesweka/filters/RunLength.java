/*
     * copyright: Anthony Bagnall
 * 
 * */
package timeseriesweka.filters;

import java.io.FileReader;

import weka.filters.*;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class RunLength extends SimpleBatchFilter {
	private int maxRunLength=50;
	private boolean useGlobalMean=true;
	private double globalMean=5.5;
	
	public RunLength(){}
	public RunLength(int maxRL){
		maxRunLength=maxRL;
	}
	public void setMaxRL(int m){
		maxRunLength=m;
	}
	public void setGlobalMean(double d){
		useGlobalMean=true;
		globalMean=d;
	}
	public void noGlobalMean(){
		useGlobalMean=false;
	}
	
	@Override
	protected Instances determineOutputFormat(Instances inputFormat)
			throws Exception {
//Treating counts as reals
		FastVector atts=new FastVector();
		Attribute a;
		for(int i=0;i<maxRunLength;i++){
			  a=new Attribute("RunLengthCount"+(i+1));
			  atts.addElement(a);
		}
		if(inputFormat.classIndex()>=0){	//Classification set, set class 
			//Get the class values as a fast vector			
			Attribute target =inputFormat.attribute(inputFormat.classIndex());
			FastVector vals=new FastVector(target.numValues());
			for(int i=0;i<target.numValues();i++)
				vals.addElement(target.value(i));
			atts.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(),vals));

		}	
		Instances result = new Instances("RunLengths"+inputFormat.relationName(),atts,inputFormat.numInstances());
		if(inputFormat.classIndex()>=0)
			result.setClassIndex(result.numAttributes()-1);
  	     return result;
	}


	@Override
	public Instances process(Instances instances) throws Exception {
		// TODO Auto-generated method stub
		Instances rl=determineOutputFormat(instances);
		if(instances.classIndex()>=0)
			rl.setClassIndex(rl.numAttributes()-1);

		Instance newInst;
		double [] d;
		for(int i=0;i<instances.numInstances();i++){

//1: Get series into an array, remove class value if present
			newInst=new DenseInstance(rl.numAttributes());
			d=instances.instance(i).toDoubleArray();
//Need to remove the class value, if present. Class value can be put at the end of the transformed data
//This needs debugging, NOT TESTED WITH CLASS INDEX < length!					
			if(instances.classIndex()>=0){	//Class has been set
				double cVal=instances.instance(i).classValue();
				double[] temp=new double[d.length-1];
				System.arraycopy(d,0,temp,0,instances.instance(i).classIndex());
				if(instances.instance(i).classIndex()<d.length-1)
					System.arraycopy(d,instances.instance(i).classIndex()+1,temp,instances.instance(i).classIndex(),d.length-instances.instance(i).classIndex()-1);
				d=temp;
			}
			
//2: Form histogram of run lengths: note missing values assumed in the same run
			double[] histogram=new double[newInst.numAttributes()];
			double t=0;
			if(useGlobalMean)
				t=globalMean;
			else{	//Find average
				int count=0;
				for(int j=0;j<d.length;j++){
					if(!instances.instance(i).isMissing(j)){
						t+=d[j];
						count++;
					}
				}
				t/=count;
			}
			int pos=1;
			int length=0;
			boolean u2=false;
			boolean under=d[0]<t?true:false;
			while(pos<d.length){
				 u2=d[pos]<t?true:false;
//					System.out.println("Pos ="+pos+" currentUNDER ="+under+"  newUNDER = "+u2);
				if(instances.instance(i).isMissing(pos)||under==u2){
					length++;
				}
				else{
//					System.out.println("Position "+pos+" has run length "+length);
					if(length<maxRunLength-1)
						histogram[length]++;
					else
						histogram[maxRunLength-1]++;
					under=u2;
					length=0;
				}
				pos++;
			}
			if(length<maxRunLength-1)
				histogram[length]++;
			else
				histogram[maxRunLength-1]++;		
			
			
/*			System.out.print("\n Histogram =  ");
			for(int k=0;k<histogram.length;k++)
				System.out.print(histogram[k]+",");
			System.out.print("\n");
*/				
//3. Put run lengths and class value into instances
			for(int j=0;j<histogram.length;j++)
				newInst.setValue(j,histogram[j]);
			if(instances.classIndex()>=0)
				newInst.setValue(rl.numAttributes()-1,instances.instance(i).classValue());
			rl.add(newInst);
		
		}
		return rl;
	}

	@Override
	public String globalInfo() {
		// TODO Auto-generated method stub
		return null;
	}
	@Override
	public String getRevision() {
		// TODO Auto-generated method stub
		return null;
	}
	
//Primitives version, assumes zero mean global, passes max run length
	public int[] processSingleSeries(double[] d, int mrl){
		double mean=0;
		int pos=1;
		int length=0;
		boolean u2=false;
		int[] histogram=new int[mrl];
		boolean under=d[0]<mean?true:false;
		while(pos<d.length){
			 u2=d[pos]<mean?true:false;
			if(under==u2){
				length++;
			}
			else{
				if(length<mrl-1)
					histogram[length]++;
				else
					histogram[mrl-1]++;
				under=u2;
				length=0;
			}
			pos++;
		}
		if(length<mrl-1)
			histogram[length]++;
		else
			histogram[mrl-1]++;		
		
		return histogram;
	}
//Test Harness
	public static void main(String[] args){
		RunLength cp=new RunLength();
		cp.noGlobalMean();
		Instances data=null;
		String fileName="C:\\Research\\Data\\Time Series Data\\Time Series Classification\\TestData\\TimeSeries_Train.arff";
		try{
			FileReader r;
			r= new FileReader(fileName); 
			data = new Instances(r); 

			data.setClassIndex(data.numAttributes()-1);
			System.out.println(data);

			Instances newInst=cp.process(data);
			System.out.println("\n"+newInst);
		}catch(Exception e)
		{
			System.out.println(" Error ="+e);
			StackTraceElement [] st=e.getStackTrace();
			for(int i=st.length-1;i>=0;i--)
				System.out.println(st[i]);
				
		}
	}
	
	
}
