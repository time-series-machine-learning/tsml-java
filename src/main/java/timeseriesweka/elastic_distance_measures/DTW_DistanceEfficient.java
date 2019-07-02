package timeseriesweka.elastic_distance_measures;
/**

Basic DTW implementation for Weka. /Each instance is assumed to be a time series.   
 **/

import java.util.Enumeration;
import weka.core.Instances;

import weka.core.neighboursearch.PerformanceStats;

public class DTW_DistanceEfficient extends DTW_DistanceBasic{
    double[] row1;
    double[] row2;
    public DTW_DistanceEfficient(){
            super();
            m_DontNormalize=true;
    }
     public DTW_DistanceEfficient(Instances data) {	
              super(data);
                    m_DontNormalize=true;
    }

 /* DTW Distance: 
  * 
  * This implementation is more memory efficient in that it only stores
  * two rows. It also implements the early abandon. If all of row 2 are above the cuttoff then 
  * we can ditch the distance
  */ 
    public double distance(double[] a,double[] b, double cutoff){
        double minDist;
        boolean tooBig=true;

    //		System.out.println("\t\t\tIn Efficient with cutoff ="+cutoff);
    // Set the longest series to a
        double[] temp;
        if(a.length<b.length){
                temp=a;
                a=b;
                b=temp;
        }
        int n=a.length;
        int m=b.length;
    //No Warp: windowSize=1, full warp: windowSize=m		
        int windowSize = getWindowSize(m);
        row1=new double[m];
        row2=new double[m];
        //Set all to max
        row1[0]=(a[0]-b[0])*(a[0]-b[0]);
        if(row1[0]<cutoff)
                tooBig=false;

        
        
        for(int j=1;j<n&&j<=windowSize;j++){
                row1[j]=Double.MAX_VALUE;
        }
    //Warp a[0] onto all b[1]...b[WindowSize]
        for(int j=1;j<windowSize && j<m;j++){
                row1[j]=row1[j-1]+(a[0]-b[j])*(a[0]-b[j]);
                if(row1[j]<cutoff)
                        tooBig=false;
        }
        if(tooBig){
                return Double.MAX_VALUE;
        }
        int start,end;

    //For each remaining row, warp row i
        for (int i=1;i<n;i++){
                tooBig=true;
                row2=new double[m];
    //Find point to start from
                if(i-windowSize<1)
                        start=0;
                else
                        start=i-windowSize+1;
                if(start==0){
                        row2[0]=row1[0]+(a[i]-b[0])*(a[i]-b[0]);
                        start=1;
                }
                else
                        row2[start-1]=Double.MAX_VALUE;
    //Find end point				
                if(start+windowSize>=m)
                        end=m;
                else
                        end=start+windowSize;
    //Warp a[i] onto b[j=start..end]
            for (int j = start;j<end;j++){
    //Find the min of row2[j-1],row1[j] and row1[j-1]
                minDist=row2[j-1];
                if(row1[j]<minDist)
                        minDist=row1[j];
                if(row1[j-1]<minDist)
                        minDist=row1[j-1];
                row2[j]=minDist+(a[i]-b[j])*(a[i]-b[j]);
                if(tooBig&&row2[j]<cutoff)
                        tooBig=false;
            }

            if(end<m)
                        row2[end]=Double.MAX_VALUE;
            //Swap row 2 into row 1.
            row1=row2;
            //Early abandon
            if(tooBig){
                return Double.MAX_VALUE;
            }
        }

        return row1[m-1];
    }

    public String toString() {
        return "DTW EFFICIENT";
    }

    /* Test Harness to check the outputs are the same with DTW Basic and TW_DistanceSpaceEfficient
    */
    public static void main(String[] args){
        DTW_DistanceBasic b=new DTW_DistanceBasic();
        DTW_DistanceEfficient c=new DTW_DistanceEfficient();
        double[] a1={1,1,1,6};
        double[] a2={1,6,6,6};
        b.setR(0);
        c.setR(0);

        System.out.println("***************** TEST 1: Two small arrays *******************");
        //Zero warp distance should be 50,
        System.out.println("\nZero warp full matrix ="+b.distance(a1,a2,Double.MAX_VALUE));
        System.out.println("Zero warp limited matrix ="+c.distance(a1,a2,Double.MAX_VALUE));
        // Full warp should be 0  		  
        b.setR(1);
        c.setR(1);
        System.out.println("\nFull warp full matrix ="+b.distance(a1,a2,Double.MAX_VALUE));
        System.out.println("Full warp limited matrix ="+c.distance(a1,a2,Double.MAX_VALUE));
        //		  System.out.println("Full warp full matrix JML version="+b.measure(a1,a2));

        // 1/4 Warp should be  25		  
        b.setR(0.25);
        c.setR(0.25);
        System.out.println("\nQuarter warp full matrix ="+b.distance(a1,a2,Double.MAX_VALUE));
        System.out.println("Quarter warp limited matrix ="+c.distance(a1,a2,Double.MAX_VALUE));


        System.out.println("***************** TEST2: Longer arrays *******************");

        //Longer arrays		  
        double[] a3={1,10,11,15,1,2,4,56,6,7,8};
        double[] a4={10,11,10,1,1,2,4,56,6,7,8};
        double d=0;
        for(int i=0;i<a3.length;i++)
              d+=(a3[i]-a4[i])*(a3[i]-a4[i]);
        System.out.println("\nEuclidean distance ="+d);	  
        //Zero warp distance should be 
        b.setR(0);
        c.setR(0);
        System.out.println("Zero warp full matrix ="+b.distance(a3,a4,Double.MAX_VALUE));
        System.out.println("Zero warp limited matrix ="+c.distance(a3,a4,100));
        //		  b.printPath();
        // Full warp should be  		  
        b.setR(1);
        c.setR(1);
        System.out.println("\nFull warp full matrix ="+b.distance(a3,a4,100));
        //		  b.printPath();
        //		  System.out.println("Full warp full matrix JML version="+b.measure(a3,a4));
        System.out.println("Full warp limited matrix ="+c.distance(a3,a4,100));
        // 1/4 Warp should be		  
        b.setR(0.25);
        c.setR(0.25);
        System.out.println("\nQuarter warp full matrix ="+b.distance(a3,a4,Double.MAX_VALUE));
        System.out.println("Quarter warp limited matrix ="+c.distance(a3,a4,Double.MAX_VALUE));
        // 1/2 Warp should be		  
        b.setR(0.5);
        c.setR(0.5);
        System.out.println("Half warp full matrix ="+b.distance(a3,a4,Double.MAX_VALUE));
        System.out.println("Half warp limited matrix ="+c.distance(a3,a4,Double.MAX_VALUE));
        //		  b.printPath();
        System.out.println("***************** TEST3: Variable length arrays *******************");

        System.out.println("NOT IMPLEMENTED FOR VARIABLE LENGTH");
        /*		  double[] a5={1,10,11};
        double[] a6={1,10,11,15,1};
        //Zero warp distance should be 50,
        System.out.println("Zero warp full matrix ="+b.distance(a5,a6,0));
        //		  System.out.println("Zero warp limited matrix ="+c.distance(a1,a2,0));
        // Full warp should be 0  		  
        b.setR(1);
        c.setR(1);
        System.out.println("Full warp full matrix ="+b.distance(a5,a6,0));
        //		  System.out.println("Full warp full matrix JML version="+b.measure(a1,a2));
        //		  System.out.println("Full warp limited matrix ="+c.distance(a1,a2,0));

        // 1/4 Warp should be  25		  
        b.setR(0.25);
        c.setR(0.25);
        System.out.println("Quarter warp full matrix ="+b.distance(a5,a6,0));
        //		  System.out.println("Quarter warp limited matrix ="+c.distance(a1,a2,0));
        */		  
        //Variable length arrays		  



    }
	  
}
