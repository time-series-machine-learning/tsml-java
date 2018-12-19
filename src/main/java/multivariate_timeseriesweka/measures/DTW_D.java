package multivariate_timeseriesweka.measures;

import static utilities.multivariate_tools.MultivariateInstanceTools.splitMultivariateInstance;
import weka.core.Instance;
import weka.core.Instances;
import timeseriesweka.elastic_distance_measures.DTW_DistanceBasic;
import weka.core.neighboursearch.PerformanceStats;


/**
 *
 * @author ABostrom
 */

public class DTW_D extends DTW_DistanceBasic{

    public DTW_D(){}
    
    public DTW_D(Instances train){
        super(train);
        
        m_Data = null;
        m_Validated = true;
    }
    
    //DIRTY HACK TO MAKE IT WORK WITH kNN. because of relational attribute stuff.
    @Override
    protected void validate() {}
    
    @Override
    public void update(Instance ins) {}
    
    @Override
    public double distance(Instance first, Instance second, double cutOffValue, PerformanceStats stats){
        //Get the double arrays
        return distance(first,second,cutOffValue);
    }
    @Override    
    public double distance(Instance first, Instance second) {
        return distance(first, second, Double.POSITIVE_INFINITY);
    }
    
    @Override
    public double distance(Instance multiSeries1, Instance multiseries2, double cutoff){
        
        //split the instance.
        Instance[] multi1 = splitMultivariateInstance(multiSeries1);
        Instance[] multi2 = splitMultivariateInstance(multiseries2);

        //TODO: might need to normalise here.
        double[][] data1 = utilities.multivariate_tools.MultivariateInstanceTools.convertMultiInstanceToTransposedArrays(multi1);
        double[][] data2 = utilities.multivariate_tools.MultivariateInstanceTools.convertMultiInstanceToTransposedArrays(multi2);
        return Math.sqrt(distance(data1, data2, cutoff));
    }
    
    
    //because a and b are transposed, we can grab a column with a[0].
    //a.length is the number of attributes
    //and a[0].length is the number of channels.
    public double distance(double[][] a, double[][] b, double cutoff){
        double minDist;
        boolean tooBig=true;
        
// Set the longest series to a
        double[][] temp;
        if(a.length<b.length){
                temp=a;
                a=b;
                b=temp;
        }
        int n=a.length;
        int m=b.length;
/*  Parameter 0<=r<=1. 0 == no warp, 1 == full warp 
generalised for variable window size
* */
        matrixD = new double[n][n];        
        windowSize = getWindowSize(n);
/*
//Set all to max. This is necessary for the window but I dont need to do 
        it all
*/

        for(int i=0;i<n;i++)
            for(int j=0;j<m;j++)
                matrixD[i][j]=Double.MAX_VALUE;

        
        matrixD[0][0]= sqMultiDist(a[0], b[0]);

//Base cases for warping 0 to all with max interval	r	
//Warp a[0] onto all b[1]...b[r+1]
        for(int j=1;j<windowSize && j<n;j++)
                matrixD[0][j]=matrixD[0][j-1]+ sqMultiDist(a[0],b[j]);

//	Warp b[0] onto all a[1]...a[r+1]
        for(int i=1;i<windowSize && i<n;i++)
                matrixD[i][0]=matrixD[i-1][0]+ sqMultiDist(a[i], b[0]);
//Warp the rest,
        for (int i=1;i<n;i++){
            tooBig=true;            
            for (int j = 1;j<m;j++){
//Find the min of matrixD[i][j-1],matrixD[i-1][j] and matrixD[i-1][j-1]
                if (i < j + windowSize && j < i + windowSize) {
                    minDist=matrixD[i][j-1];
                    if(matrixD[i-1][j]<minDist)
                            minDist=matrixD[i-1][j];
                    if(matrixD[i-1][j-1]<minDist)
                            minDist=matrixD[i-1][j-1];
                    matrixD[i][j]=minDist + sqMultiDist(a[i], b[j]);
                    if(tooBig&&matrixD[i][j]<cutoff)
                            tooBig=false;               
                }
            }
            //Early abandon
            if(tooBig){
                return Double.MAX_VALUE;
            }
            
        }			
//Find the minimum distance at the end points, within the warping window. 
        return matrixD[n-1][m-1];
    }
    
    
    double sqDist(double a, double b){
        return (a-b)*(a-b);
    }
    
    //given each aligned value in the channel.
    double sqMultiDist(double[] a, double[] b){
        double sum = 0;
        for(int i=0; i<a.length; i++){
            sum += sqDist(a[i], b[i]);
        }
        return sum;
    }
    
    
    public static void main(String[] args)
    {

    }
	
}
