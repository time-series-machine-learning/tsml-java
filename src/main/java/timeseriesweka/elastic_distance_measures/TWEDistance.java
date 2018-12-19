/*

* Java distance measure derived from
* Filename: code-twed.c
source code for the Time Warp Edit Distance in ansi C.
Author: Pierre-Francois Marteau
Version: V1.1 du 10/3/2010
Licence: GPL
******************************************************************
This software and description is free delivered "AS IS" with no
guaranties for work at all. Its up to you testing it modify it as
you like, but no help could be expected from me due to lag of time
at the moment. I will answer short relevant questions and help as
my time allow it. I have tested it played with it and found no
problems in stability or malfunctions so far.
Have fun.
*****************************************************************
Please cite as:
@article{Marteau:2009:TWED,
 author = {Marteau, Pierre-Francois},
 title = {Time Warp Edit Distance with Stiffness Adjustment for Time Series 
Matching},
 journal = {IEEE Trans. Pattern Anal. Mach. Intell.},
 issue_date = {February 2009},
 volume = {31},
 number = {2},
 month = feb,
 year = {2009},
 issn = {0162-8828},
 pages = {306--318},,
}
* Original code was structured to work on a set of time series and
* had an extra parameter for the power of the pointwise distance measure
* We implement it pairwise and assume Euclidean distance, for simplicity
* and equivalence to the other measures.
*/


package timeseriesweka.elastic_distance_measures;

import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.neighboursearch.PerformanceStats;


/*
 * TWE has two parameters: double nu and lambda
 * nu controls the "stiffness". It is a multiplicative penalty
 * on the distance between matched points. 0 gives no weighting (full DTW),
 * infinity gives Euclidean distance.
 * lambda is a constant penalty for the
 * amount of shrinkage caused be the delete operation.
* Ranges for parameters are 0... infinity.
 *
 * from the paper: "stiffness value nu is selected from
 * 10^-5; 10^-4; 10^-3; 10^-2; 10^-1; 1 and 
lambda is selected from 0; .25; .5; .75; 1.0.
 */


public class TWEDistance extends EuclideanDistance{

    double nu=1;
    double lambda=1;
    double degree=2;
    public void setNu(double n){nu=n;}
    public void setLambda(double n){lambda=n;}


    public TWEDistance(){
        super();
        this.m_DontNormalize = true;
    }
    
    public TWEDistance(double nu, double lambda){
        super();
        this.m_DontNormalize = true;
        this.nu = nu;
        this.lambda = lambda;
    }


    /**
     * Distance method
     *
     * @param first instance 1
     * @param second instance 2
     * @param cutOffValue used for early abandon
     * @param stats
     * @return distance between instances
     */
    @Override
    public double distance(Instance first, Instance second, 
            double cutOffValue, PerformanceStats stats){
        //Get the double arrays
        return distance(first,second,cutOffValue);
    }

    /**
     * distance method that converts instances to arrays of doubles
     *
     * @param first instance 1
     * @param second instance 2
     * @param cutOffValue used for early abandon
     * @return distance between instances
     */
    @Override
    public double distance(Instance first, Instance second, 
            double cutOffValue)
    {
        //remove class index from first instance if there is one
        int firtClassIndex = first.classIndex();
        double[] arr1;
        if(firtClassIndex > 0){
            arr1 = new double[first.numAttributes()-1];
            for(int i = 0,j = 0; i < first.numAttributes(); i++){
                if(i != firtClassIndex){
                    arr1[j]= first.value(i);
                    j++;
                }
            }
        }else{
            arr1 = first.toDoubleArray();
        }

        //remove class index from second instance if there is one
        int secondClassIndex = second.classIndex();
        double[] arr2;
        if(secondClassIndex > 0){
            arr2 = new double[second.numAttributes()-1];
            for(int i = 0,j = 0; i < second.numAttributes(); i++){
                if(i != secondClassIndex){
                    arr2[j]= second.value(i);
                    j++;
                }
            }
        }else{
            arr2 = second.toDoubleArray();
        }

        return distance(arr1,arr2,cutOffValue);
    }

    /**
     * calculates the distance between two instances (been converted to arrays)
     * Altered c code from the authors downloaded from
     *
     * http://www-irisa.univ-ubs.fr/Pierre-Francois.Marteau/TWED/code-twed.c
     * @param first instance 1 as array
     * @param second instance 2 as array
     *
     * @return distance between instances
     */
	public double TWE_Distance(double[] a, double[] b){
/*This code is faithful to the c version, so uses a redundant
 * Multidimensional representation. The c code does not describe what the 
            arguments
 * tsB and tsA are. We assume they are the time stamps (i.e. index sets),
 * and initialise them accordingly.
 */

            int dim=1;
            double dist, disti1, distj1;
            double[][] ta=new double[a.length][dim];
            double[][] tb=new double[a.length][dim];
            double[] tsa=new double[a.length];
            double[] tsb=new double[b.length];
            for(int i=0;i<tsa.length;i++)
                tsa[i]=(i+1);
            for(int i=0;i<tsb.length;i++)
                tsb[i]=(i+1);

            int r = ta.length;
            int c = tb.length;
	int i,j,k;
//Copy over values
        for(i=0;i<a.length;i++)
            ta[i][0]=a[i];
        for(i=0;i<b.length;i++)
            tb[i][0]=b[i];

        /* allocations in c
	double **D = (double **)calloc(r+1, sizeof(double*));
	double *Di1 = (double *)calloc(r+1, sizeof(double));
	double *Dj1 = (double *)calloc(c+1, sizeof(double));
	for(i=0; i<=r; i++) {
		D[i]=(double *)calloc(c+1, sizeof(double));
	}
*/
	double [][]D = new double[r+1][c+1];
	double[] Di1 = new double[r+1];
	double[] Dj1 = new double[c+1];
// local costs initializations
	for(j=1; j<=c; j++) {
  		distj1=0;
                for(k=0; k<dim; k++)
      		  if(j>1){
//CHANGE AJB 8/1/16: Only use power of 2 for speed up,                       
                    distj1+=(tb[j-2][k]-tb[j-1][k])*(tb[j-2][k]-tb[j-1][k]);
// OLD VERSION                    distj1+=Math.pow(Math.abs(tb[j-2][k]-tb[j-1][k]),degree);
// in c:               distj1+=pow(fabs(tb[j-2][k]-tb[j-1][k]),degree);
             	  }
             	  else
              		distj1+=tb[j-1][k]*tb[j-1][k];                      
//OLD              		distj1+=Math.pow(Math.abs(tb[j-1][k]),degree);
   		Dj1[j]=(distj1);
	}

	for(i=1; i<=r; i++) {
            disti1=0;
            for(k=0; k<dim; k++)
              if(i>1)
                disti1+=(ta[i-2][k]-ta[i-1][k])*(ta[i-2][k]-ta[i-1][k]);
// OLD                 disti1+=Math.pow(Math.abs(ta[i-2][k]-ta[i-1][k]),degree);
              else 
                  disti1+=(ta[i-1][k])*(ta[i-1][k]);
//OLD                  disti1+=Math.pow(Math.abs(ta[i-1][k]),degree);

            Di1[i]=(disti1);

            for(j=1; j<=c; j++) {
                dist=0;
                for(k=0; k<dim; k++){
                  dist+=(ta[i-1][k]-tb[j-1][k])*(ta[i-1][k]-tb[j-1][k]);
//                  dist+=Math.pow(Math.abs(ta[i-1][k]-tb[j-1][k]),degree);
                  if(i>1&&j>1)
                    dist+=(ta[i-2][k]-tb[j-2][k])*(ta[i-2][k]-tb[j-2][k]);                      
//                    dist+=Math.pow(Math.abs(ta[i-2][k]-tb[j-2][k]),degree);
                }
                D[i][j]=(dist);
            }
	}// for i

	// border of the cost matrix initialization
	D[0][0]=0;
	for(i=1; i<=r; i++)
  		D[i][0]=D[i-1][0]+Di1[i];
	for(j=1; j<=c; j++)
  		D[0][j]=D[0][j-1]+Dj1[j];

	double dmin, htrans, dist0;
	int iback;

	for (i=1; i<=r; i++){
            for (j=1; j<=c; j++){
                htrans=Math.abs((tsa[i-1]-tsb[j-1]));
                if(j>1&&i>1)
                        htrans+=Math.abs((tsa[i-2]-tsb[j-2]));
                dist0=D[i-1][j-1]+nu*htrans+D[i][j];
                dmin=dist0;
                if(i>1)
                        htrans=((tsa[i-1]-tsa[i-2]));
                  else htrans=tsa[i-1];
                dist=Di1[i]+D[i-1][j]+lambda+nu*htrans;
                if(dmin>dist){
                    dmin=dist;
                }
                if(j>1)
                        htrans=(tsb[j-1]-tsb[j-2]);
                  else htrans=tsb[j-1];
                dist=Dj1[j]+D[i][j-1]+lambda+nu*htrans;
                if(dmin>dist){
                    dmin=dist;
                }
                D[i][j] = dmin;
            }
	}

	dist = D[r][c];
        return dist;
}






    public double distance(double[] first, double[] second,
            double cutOffValue){
        return TWE_Distance(first,second);
    }


}