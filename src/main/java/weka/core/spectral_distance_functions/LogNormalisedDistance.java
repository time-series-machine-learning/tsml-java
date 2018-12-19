/*
 *Log normalised distance function for the periodogram as proposed in
 * 
 * @ARTICLE{caiado06periodogram,
    author = {J. CAIADO and N. CRATO and D. PEÑA },
    title = {A periodogram-based metric for time series classification},
    journal = {Computational Statistics & Data Analysis},
    year = {2006},
    volume = {50},
    pages = {2668--2684}
}
 Note that Caiado et al use the normalised version of the periodogram. 
 * This distance function is directly linked to the ACF, in that 
 * d_NP(x,y) = (2 sqrt(n) d_ACF(x,y).
 * However, different results may occur because of different truncations.
 * 
  .
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
public class LogNormalisedDistance extends EuclideanDistance{
	private static final long serialVersionUID = 1L;
	public LogNormalisedDistance(){
		super();
	}
	  public LogNormalisedDistance(Instances data) {	
		  super(data);
	  }
	
	//Needs overriding to avoid cutoff check
    @Override
	  public double distance(Instance first, Instance second){
		  return distance(first, second, Double.POSITIVE_INFINITY, null, false);
	  }
    @Override
	  public double distance(Instance first, Instance second, PerformanceStats stats) { //debug method pls remove after use
		    return distance(first, second, Double.POSITIVE_INFINITY, stats, false);
		  }

	  
    @Override
	  public double distance(Instance first, Instance second, double cutOffValue, PerformanceStats stats){
		  return distance(first,second,cutOffValue,stats,false);
		  }
	  public double distance(Instance first, Instance second, double cutOffValue, PerformanceStats stats, boolean print) {
		  	//Get the double arrays
		  return distance(first,second,cutOffValue);
	  }
	  
    @Override
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
                    dist+=Math.log(a[i])-Math.log(b[i]);
                        if(Math.sqrt(dist)>cutoff)
                            return Double.POSITIVE_INFINITY;
		}
		return Math.sqrt(dist);
	  }

    @Override
	  public String toString() {
		    return "Log normalised distance";
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
