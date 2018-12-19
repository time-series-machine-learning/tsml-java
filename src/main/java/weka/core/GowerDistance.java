package weka.core;

import weka.core.neighboursearch.PerformanceStats;

public class GowerDistance extends EuclideanDistance {
	private static final long serialVersionUID = 1L;
	  double[] ranges;

	
	private GowerDistance(){
		super();
	}
	  public GowerDistance(Instances data) {
		  super(data);
		  ranges=findRanges(data);
	  }
          
	  double[] findRanges(Instances d){
		  int classIndex=d.classIndex();
		  double[] r;
		  if(classIndex>0)
			  r= new double[d.numAttributes()-1];
		  else
			  r= new double[d.numAttributes()];
		  int c=0;
		  for(int j=0;j<d.numAttributes();j++){
			 if(j!=classIndex){
				 //Find min and max range for attribute i
				 double min=d.instance(0).value(0);
				 double max=min;
				  for(int i=1;i<d.numInstances();i++){
					  double x=d.instance(i).value(j);
						 if(x>max)
							 max=x;
						 if(x<min)
							 min=x;
				  }
				  //Set range as max-min
				  r[c]=max-min;
				  c++;	
			 }
		  }
		  return r;
	  }
	  
	//Needs overriding to avoid cutoff check
	  public double distance(Instance first, Instance second){
		  return distance(first, second, Double.POSITIVE_INFINITY, null, false);
	  }
	  public double distance(Instance first, Instance second, PerformanceStats stats) { //debug method pls remove after use
		    return distance(first, second, Double.POSITIVE_INFINITY, stats, false);
		  }
	  public double distance(Instance first, Instance second, double cutOffValue, PerformanceStats stats){
		  return distance(first,second,cutOffValue,stats,false);
		  }
	  public double distance(Instance first, Instance second, double cutOffValue, PerformanceStats stats, boolean print) {
		  return distance(first,second,cutOffValue);
	  }
  
	  
	  public double distance(Instance first, Instance second, double cutOffValue) {
		  	//Get the double arrays without the class value.
		  int classIndex=first.classIndex();
		  double[] f1,f2;
		  if(classIndex>=0){
			  f1=new double[first.numAttributes()-1];
			  f2=new double[second.numAttributes()-1];
			  int c=0;
			  for(int i=0;i<first.numAttributes();i++){
				  if(i!=classIndex){
					  f1[c]=first.value(i);
					  f2[c]=second.value(i);
					  c++;
				  }
			  }
		  }else{
			  f1=new double[first.numAttributes()];
			  f2=new double[second.numAttributes()];
			  for(int i=0;i<first.numAttributes();i++){
				  f1[i]=first.value(i);
				  f2[i]=second.value(i);
			  }
		  }
		  return distance(f1,f2,cutOffValue);
	  }
/** This is the method that actually does the calculation. 
 * Only implemented for ordinal/real valued variables
 * @param a
 * @param b
 * @param cutoff
 * @return
 */
	  public double distance(double[] a,double[] b, double cutoff){
		  
		  
		  // 1. Work out the ranges for each variable
		  if(ranges==null){
			  System.out.println("Error in Gower distance, ranges not calculated, exiting but should throw an exception!");
			  System.exit(0);
		  }
		  // 2. 
//For all real attributes, 		  
		  double dist=0;
		  for(int i=0;i<a.length;i++){
			  dist+=Math.abs(a[i]-b[i])/ranges[i];
		  }
		  return dist;
	  }
}
