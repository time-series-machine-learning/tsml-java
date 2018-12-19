/*
DTW with early abandon
 */
package timeseriesweka.elastic_distance_measures;

/**
 *
 * @author ajb
 */
public final class DTW extends DTW_DistanceBasic {
    
    /**
     *
     * @param a
     * @param b
     * @param cutoff
     * @return
     */
    @Override
 public final double distance(double[] a,double[] b, double cutoff){
        double minDist;
        boolean tooBig;
// Set the longest series to a. is this necessary?
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
        windowSize = getWindowSize(n);
//Extra memory than required, could limit to windowsize,
//        but avoids having to recreate during CV 
//for varying window sizes        
        if(matrixD==null)
            matrixD=new double[n][m];
        
/*
//Set boundary elements to max. 
*/
        int start,end;
        for(int i=0;i<n;i++){
            start=windowSize<i?i-windowSize:0;
            end=i+windowSize+1<m?i+windowSize+1:m;
            for(int j=start;j<end;j++)
                matrixD[i][j]=Double.MAX_VALUE;
        }
        matrixD[0][0]=(a[0]-b[0])*(a[0]-b[0]);
//a is the longer series. 
//Base cases for warping 0 to all with max interval	r	
//Warp a[0] onto all b[1]...b[r+1]
        for(int j=1;j<windowSize && j<m;j++)
                matrixD[0][j]=matrixD[0][j-1]+(a[0]-b[j])*(a[0]-b[j]);

//	Warp b[0] onto all a[1]...a[r+1]
        for(int i=1;i<windowSize && i<n;i++)
                matrixD[i][0]=matrixD[i-1][0]+(a[i]-b[0])*(a[i]-b[0]);
//Warp the rest,
        for (int i=1;i<n;i++){
            tooBig=true; 
            start=windowSize<i?i-windowSize+1:1;
            end=i+windowSize<m?i+windowSize:m;
            for (int j = start;j<end;j++){
                    minDist=matrixD[i][j-1];
                    if(matrixD[i-1][j]<minDist)
                            minDist=matrixD[i-1][j];
                    if(matrixD[i-1][j-1]<minDist)
                            minDist=matrixD[i-1][j-1];
                    matrixD[i][j]=minDist+(a[i]-b[j])*(a[i]-b[j]);
                    if(tooBig&&matrixD[i][j]<cutoff)
                            tooBig=false;               
            }
            //Early abandon
            if(tooBig){
                return Double.MAX_VALUE;
            }
        }			
//Find the minimum distance at the end points, within the warping window. 
        return matrixD[n-1][m-1];
    }
        
}
