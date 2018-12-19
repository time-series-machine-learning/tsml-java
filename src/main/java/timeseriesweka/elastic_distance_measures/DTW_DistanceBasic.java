package timeseriesweka.elastic_distance_measures;
/**

Basic DTW implementation for Weka. /Each instance is assumed to be a time series. Basically we
pull all the data out and proceed as usual!
  
 **/

import java.util.ArrayList;
import java.util.Enumeration;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;

import weka.core.neighboursearch.PerformanceStats;

public class DTW_DistanceBasic extends EuclideanDistance{

    private static final long serialVersionUID = 1L;
    protected int windowSize;
    protected double r=1;	//Warping window size percentage, between 0 and 1
    protected double[][] matrixD;
    protected int endX=0;
    protected int endY=0;
    public DTW_DistanceBasic(){
            super();
            m_DontNormalize=true;

    }
    public DTW_DistanceBasic(Instances data) {	
        super(data);
        m_DontNormalize=true;
    }
    //Needs overriding to avoid cutoff check
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
        return distance(first,second,cutOffValue);
    }
  
    private double[] extractSeries(Instance a){
        int fClass=a.classIndex();
        double[] s;
        if(fClass>0) {
            s=new double[a.numAttributes()-1];
            int count=0;
            for(int i=0;i<s.length+1;i++){
                if(i!=fClass){
                    s[count]=a.value(i);
                    count++;
                }
            }
        }
        else
            s=a.toDoubleArray();
        return s;
    }
    @Override
    public double distance(Instance first, Instance second, double cutOffValue) {
        double[] f=extractSeries(first);
        double[] s=extractSeries(second);
        return distance(f,s,cutOffValue);
    }

    /* DTW Distance with early abandon: 
    * 
    */ 
    public double distance(double[] a,double[] b, double cutoff){
        double minDist;
        boolean tooBig=true;
        
// Set the longest series to a
        double[] temp;
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

        
        matrixD[0][0]=(a[0]-b[0])*(a[0]-b[0]);

//Base cases for warping 0 to all with max interval	r	
//Warp a[0] onto all b[1]...b[r+1]
        for(int j=1;j<windowSize && j<n;j++)
                matrixD[0][j]=matrixD[0][j-1]+(a[0]-b[j])*(a[0]-b[j]);

//	Warp b[0] onto all a[1]...a[r+1]
        for(int i=1;i<windowSize && i<n;i++)
                matrixD[i][0]=matrixD[i-1][0]+(a[i]-b[0])*(a[i]-b[0]);
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
                    matrixD[i][j]=minDist+(a[i]-b[j])*(a[i]-b[j]);
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

    static public int findWindowSize(double rr,int n){
        int w=(int)(rr*n);   //Rounded down.
                //No Warp, windowSize=1
        if(w<1) w=1;
                //Full Warp : windowSize=n, otherwise scale between		
        else if(w<n)    
                w++;
        return w;	
    }    
     
    final public int getWindowSize(int n){
        int w=(int)(r*n);   //Rounded down.
                //No Warp, windowSize=1
        if(w<1) w=1;
                //Full Warp : windowSize=n, otherwise scale between		
        else if(w<n)    
                w++;
        return w;	
    }
    final public int findMaxWindow(){
        //Find Path backwards in pairs			
        int n=matrixD.length;
        int m=matrixD[0].length;
        int x=n-1,y=m-1;
        int maxDiff=0;
        while(x>0 && y>0)
        {
            //Look along
            double min=matrixD[x-1][y-1];
            if(min<=matrixD[x-1][y] && min<=matrixD[x][y-1]){
                    x--;
                    y--;
            }
            else if(matrixD[x-1][y] < matrixD[x][y-1])
                    x--;
            else
                    y--;
            int diff=(x>y)?x-y:y-x;
            if(diff>maxDiff)
                maxDiff=diff;
        }
        return maxDiff;
        
    }
    void printPath(){
        //Find Path backwards in pairs			
        int n=matrixD.length;
        int m=matrixD[0].length;
        int x=n-1,y=m-1;
        int count=0;
        System.out.println(count+"END  Point  = "+x+","+y+" value ="+matrixD[x][y]);
        while(x>0 && y>0)
        {
            //Look along
            double min=matrixD[x-1][y-1];
            if(min<=matrixD[x-1][y] && min<=matrixD[x][y-1]){
                    x--;
                    y--;
            }
            else if(matrixD[x-1][y] < matrixD[x][y-1])
                    x--;
            else
                    y--;
            count++;
            System.out.println(count+" Point  = "+x+","+y+" value ="+matrixD[x][y]);
        }
        while(x>0){
            x--;
            System.out.println(count+" Point  = "+x+","+y+" value ="+matrixD[x][y]);
        }
        while(y>0){
            y--;
            System.out.println(count+" Point  = "+x+","+y+" value ="+matrixD[x][y]);
        }
    }
    public String toString() { return "DTW BASIC. r="+r;}
    public String globalInfo() {return " DTW Basic Distance";}
    public String getRevision() {return "Version 1.0";  }
    public void setR(double x){ r=x;}
    public double getR(){ return r;}
    public int getWindowSize(){ return windowSize;}



    public static void main(String[] args)
    {
            System.out.println(" Very basic test for DTW distance");
            double[] a ={1,2,3,4,5,6,7,8};
            double[] b ={2,3,4,5,6,7,8,9};
            for(int i=0;i<a.length;i++)
                    System.out.print(a[i]+",");
            System.out.println("\n************");
            for(int i=0;i<b.length;i++)
                    System.out.print(b[i]+",");
            System.out.println("\n Euclidean distance is 8, DTW should be 2");
            DTW_DistanceBasic dtw= new DTW_DistanceBasic();
 //           dtw.printPath();
/*            for(double[] d:dtw.matrixD){
                for(double x:d)
                    System.out.print(x+",");
                System.out.print("\n");
          }
  */                


    }
	
}
