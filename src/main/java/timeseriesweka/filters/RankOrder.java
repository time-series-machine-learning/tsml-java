/*
     * copyright: Anthony Bagnall
 * */
package timeseriesweka.filters;

import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.Capabilities.Capability;
import weka.core.matrix.*;
import weka.filters.*;

import java.util.*;
import weka.core.DenseInstance;

public class RankOrder extends SimpleBatchFilter {

	public double[][] ranks;
		public int numAtts=0;
		private boolean normalise=true;
		public void setNormalise(boolean f){normalise=f;};  
		  
	  public Instances process(Instances inst) throws Exception {
//Set input instance format		  
		 
		  Instances result = new Instances(determineOutputFormat(inst), 0);
		  rankOrder(inst);
//Stuff into new set of instances
		  for(int i=0;i<inst.numInstances();i++) {
//Create a deep copy, think this is necessary to maintain meta data?
			  Instance in=new DenseInstance(inst.instance(i)); 
//Reset to the ranks
			  for(int j=0;j<numAtts;j++)
				  in.setValue(j, ranks[i][j]);
			  result.add(in);
		  }
		  if(normalise){
			  NormalizeAttribute na=new NormalizeAttribute(result);
			  result=na.process(result);
		  }
		  return result;

	  }
	  protected class Pair implements Comparable{
		  int pos;
		  double val;
		  public Pair(int p, double d){
			  pos=p;
			  val=d;
		  }
		  public int compareTo(Object c){
			  if(val>((Pair)c).val) return 1;
			  if(val<((Pair)c).val) return -1;
			  return 0;
		  }
		  
	  }

	  
	  
	  public void rankOrder(Instances inst){  
		  numAtts=inst.numAttributes();
		  int c= inst.classIndex();
		  if(c>0)
			  numAtts--;
	//If a classification problem it is assumed the class attribute is the last one in the instances
 		  Pair[][] d =new Pair[numAtts][inst.numInstances()];
		  for(int j=0;j<inst.numInstances();j++){
				Instance x=inst.instance(j);
				for(int i=0;i<numAtts;i++)
					d[i][j]=new Pair(j,x.value(i));
		  }
		  //Form rank order data set (in transpose of sorted array)
			  //Sort each array of Pair
			for(int i=0;i<numAtts;i++)
				Arrays.sort(d[i]);
			ranks=new double[inst.numInstances()][numAtts];
			for(int j=0;j<inst.numInstances();j++)
				for(int i=0;i<numAtts;i++)	
					ranks[d[i][j].pos][i]=j;

		  }
	  
	  public static void testFilter(Instances data, Filter ct){
			try{
				data.deleteStringAttributes();
				ct.setInputFormat(data);
				Instances newData=Filter.useFilter(data,ct);
				System.out.print(newData);
			}catch(Exception e){
				System.err.println("Exception thrown ="+e);
				System.err.println("Stack =");
				StackTraceElement[] str=e.getStackTrace();
				for(StackTraceElement s:str)
					System.err.println(s);
			}	
		  
	  }
	
			@Override
			protected Instances determineOutputFormat(Instances inputFormat){
			     Instances result = new Instances(inputFormat, 0);
			     return result;
			}

			@Override
			public String globalInfo() {
				return null;
			}

			public Capabilities getCapabilities() {
				 Capabilities result = super.getCapabilities();
				result.enableAllAttributes();
				 result.enableAllClasses();
				 result.enable(Capability.NO_CLASS);  // filter doesn't need class to be set
				return result;
				}
			public String getRevision() {
				// TODO Auto-generated method stub
				return null;
			}  
	
			public static void main(String[] args){
				
			}

}
