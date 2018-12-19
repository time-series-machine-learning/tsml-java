/*
  * K-L distance for periodograms as defined in
  * 
 * @ARTICLE{caiado06periodogram,
    author = {J. CAIADO and N. CRATO and D. PEÑA },
    title = {A periodogram-based metric for time series classification},
    journal = {Computational Statistics & Data Analysis},
    year = {2006},
    volume = {50},
    pages = {2668--2684}
}
* 
where it is stated that the K-L distance in the spectral domain is asymptotically
* equivalent to that in the frequence domain.

*/
package weka.core.spectral_distance_functions;

import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.PerformanceStats;
import weka.filters.NormalizeCase;

/**
 *
 * @author ajb
 */
public class KullbackLeiberDistance extends EuclideanDistance{
	private static final long serialVersionUID = 1L;
	public KullbackLeiberDistance(){
		super();
	}
	  public KullbackLeiberDistance(Instances data) {	
		  super(data);
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
		  	//Get the double arrays
		  return distance(first,second,cutOffValue);
	  }
	  
public double distance(Instance first, Instance second, double cutOffValue) {
		  
		  double[] f;
		  double[] s;
		  int fClass=first.classIndex();
		  if(fClass>=0) {
			  f=new double[first.numAttributes()-1];
			  int count=0;
			  for(int i=0;i<f.length+1;i++){
				  if(i!=fClass){
					  f[count]=first.value(i);
					  count++;
				  }
			  }
		  }
		  else
			  f=first.toDoubleArray();
		  int sClass=second.classIndex();
		  if(sClass>=0) {
			  s=new double[second.numAttributes()-1];
			  int count=0;
			  for(int i=0;i<s.length;i++){
				  if(i!=sClass){
					  s[count]=second.value(i);
					  count++;
				  }
			  }
		  }
		  else
			  s=second.toDoubleArray();
		  if(f.length!=s.length){	//Error here 
			  System.out.println("Error in distance calculation for Likelihhod ratio, unequal lengths, exiting program!");
			  System.exit(0);
		  }
                                    if(!m_DontNormalize){ //If the series have been pre normalised, there is no need to do this. 
                      try{
                        NormalizeCase.standardNorm(f);
                        NormalizeCase.standardNorm(s);
                      }catch(Exception e){
                          System.out.println(" in log norm distance, Exception ="+e);
                          e.printStackTrace();
                          System.exit(0);
                      }
                          
                  }
		  return distance(f,s,cutOffValue);
	  }

	  /* KL Distance distance
	  * 
	  */ 
	  public double distance(double[] a,double[] b, double cutoff){
		double dist=0;
		for(int i=0;i<a.length;i++){
                    dist+=a[i]/b[i]-Math.log(a[i])/Math.log(b[i])-1;
                        if(dist>cutoff)
                            return Double.POSITIVE_INFINITY;
		}
		return dist;
	  }

	  public String toString() {
		    return "Likelihood Ratio";
	  }
	public String globalInfo() {
		return "Likelihood Ratio";
	}
	@Override
	protected double updateDistance(double currDist, double diff) {
		// TODO Auto-generated method stub
		return 0;
	}
	public String getRevision() {
	
		return null;
	}
	
    
}
