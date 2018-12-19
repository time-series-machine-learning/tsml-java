package weka.core.spectral_distance_functions;
/**


  
 **/

import java.util.ArrayList;
import java.util.Enumeration;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;

import weka.core.neighboursearch.PerformanceStats;

public class LikelihoodRatioDistance extends EuclideanDistance{
	private static final long serialVersionUID = 1L;
	public LikelihoodRatioDistance(){
		super();
	}
	  public LikelihoodRatioDistance(Instances data) {	
		  super(data);
	  }
	
	//Needs overriding to avoid cutoff check
	  public double distance(Instance first, Instance second){
		  return distance(first, second, Double.POSITIVE_INFINITY, null, false);
	  }
	  public double distance(Instance first, Instance second, PerformanceStats stats) { //debug method pls remove after use
		    return distance(first, second, Double.POSITIVE_INFINITY, stats, false);
		  }

	  
/* ASSUMES THE CLASS INDEX IS THE LAST DATA FOR THE NORMALISATION. But dont do the normalisation here anyway.
 * 
	Euclidean distance normalises to [0..1] based on the flag m_DontNormalize, but it does this in the calculation.
More efficient to do this as a filter, otherwise you are repeatedly recalculating.Basic normalisation is implemented, but advised 
not to use it, so flag set by default to true. 

*/
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

		  return distance(f,s,cutOffValue);
	  }

	  /* Likelihood ratio distance
	  * 
	  */ 
	  public double distance(double[] a,double[] b, double cutoff){
		double dist=0;
		double n1,n2,n;
		n1=a[0];
		for(int i=1;i<a.length;i++)
			n1+=a[i];
		n2=b[0];
		for(int i=1;i<b.length;i++)
			n2+=b[i];
		n=n1+n2;
		for(int i=0;i<a.length;i++){
			if(a[i]>0)
				dist+=(a[i]/n1)*Math.log((a[i]/n1)/((a[i]+b[i])/n));
			if(b[i]>0)
				dist+=(b[i]/n2)*Math.log((b[i]/n2)/((a[i]+b[i])/n));
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
	
/* JML Implementation version
 * 
 */
 	
}
