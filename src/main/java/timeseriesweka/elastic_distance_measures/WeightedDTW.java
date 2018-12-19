/*
 * A simple DTW algorithm that computes the warped path with no constraints 
 * 
 */
package timeseriesweka.elastic_distance_measures;

import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.PerformanceStats;

/**
 *
 * @author Jason Lines (adapted from BasicDTW)
 */
public class WeightedDTW extends BasicDTW{
    
    protected double[][] distances;
    protected boolean isEarlyAbandon;

    private static final double WEIGHT_MAX = 1;
    private double g; // "empirical constant that controls the curvature (slope) of the function
    private double[] weightVector; // initilised on first distance call

//    private int distanceCount = 0;
    
        
    /**
     * BasicDTW Constructor 
     * 
     * Early Abandon Disabled
     */
    public WeightedDTW(){
        super();
        this.g = 0;
        this.weightVector = null;
        this.m_DontNormalize = true;
        this.isEarlyAbandon = false;
    }

    public WeightedDTW(double g){
        super();
        this.g = g;
        this.weightVector = null;
        this.m_DontNormalize = true;
        this.isEarlyAbandon = false;
    }

    public WeightedDTW(double g, double[] weightVector){
        super();
        this.g = g;
        this.weightVector = weightVector;
        this.m_DontNormalize = true;
        this.isEarlyAbandon = false;
    }
    
    /** 
     * BasicDTW Constructor that allows enabling of early abandon
     * 
     * @param earlyAbandon boolean value setting if early abandon is enabled
     */
    public WeightedDTW(boolean earlyAbandon) {	
        super();
        this.g = 0;
        this.weightVector = null;
        this.isEarlyAbandon = earlyAbandon;
        this.m_DontNormalize = true;
    }

    public WeightedDTW(double g, boolean earlyAbandon) {
        super();
        this.g = g;
        this.weightVector = null;
        this.isEarlyAbandon = earlyAbandon;
        this.m_DontNormalize = true;
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
    public double distance(Instance first, Instance second, double cutOffValue, PerformanceStats stats){
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
    public double distance(Instance first, Instance second, double cutOffValue){
//        if(this.distanceCount % 10000000 == 0){
//            System.out.println("New Instance: "+this.distanceCount);
//        }
//        this.distanceCount++;
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
     * 
     * @param first instance 1 as array
     * @param second instance 2 as array
     * @param cutOffValue used for early abandon
     * @return distance between instances
     */
    public double distance(double[] first, double[] second, double cutOffValue){

        if(this.weightVector==null){
            this.initWeights(first.length);
        }


        //create empty array
        this.distances = new double[first.length][second.length];
        
        //first value
        this.distances[0][0] = this.weightVector[0]*(first[0]-second[0])*(first[0]-second[0]);
        
        //early abandon if first values is larger than cut off
        if(this.distances[0][0] > cutOffValue && this.isEarlyAbandon){
            return Double.MAX_VALUE;
        }
        
        //top row
        for(int i=1;i<second.length;i++){
            this.distances[0][i] = this.distances[0][i-1]+this.weightVector[i]*(first[0]-second[i])*(first[0]-second[i]); //edited by Jay
        }

        //first column
        for(int i=1;i<first.length;i++){
            this.distances[i][0] = this.distances[i-1][0]+this.weightVector[i]*(first[i]-second[0])*(first[i]-second[0]); //edited by Jay
        }
        
        //warp rest
        double minDistance;
        for(int i = 1; i<first.length; i++){
            boolean overflow = true;
            
            for(int j = 1; j<second.length; j++){
                //calculate distances
                minDistance = Math.min(this.distances[i][j-1], Math.min(this.distances[i-1][j], this.distances[i-1][j-1]));
                this.distances[i][j] = minDistance+this.weightVector[Math.abs(i-j)] *(first[i]-second[j])*(first[i]-second[j]); //edited by Jay
                
                if(overflow && this.distances[i][j] < cutOffValue){
                    overflow = false; // because there's evidence that the path can continue
                }
//                    
//                if(minDistance > cutOffValue && this.isEarlyAbandon){
//                    this.distances[i][j] = Double.MAX_VALUE;
//                }else{
//                    this.distances[i][j] = minDistance+this.weightVector[Math.abs(i-j)] *(first[i]-second[j])*(first[i]-second[j]); //edited by Jay
//                    overflow = false;
//                }
            }
            
            //early abandon
            if(overflow && this.isEarlyAbandon){
                return Double.MAX_VALUE;
            }
        }
        return this.distances[first.length-1][second.length-1];
    }

//    // Added by Jay for weighted DTW
//    private double getWeight(int seriesLength, int i, int j){
//        return WEIGHT_MAX/(1+Math.exp(-g*(Math.abs(i-j)-(double)seriesLength/2)));
//    }

    private void initWeights(int seriesLength){
        this.weightVector = new double[seriesLength];
        double halfLength = (double)seriesLength/2;

        for(int i = 0; i < seriesLength; i++){
            weightVector[i] = WEIGHT_MAX/(1+Math.exp(-g*(i-halfLength)));
        }
    }

    public static double[] calculateWeightVector(int seriesLength, double g){
        double[] weights = new double[seriesLength];
        double halfLength = (double)seriesLength/2;

        for(int i = 0; i < seriesLength; i++){
            weights[i] = WEIGHT_MAX/(1+Math.exp(-g*(i-halfLength)));
        }
        return weights;
    }




    /**
     * Generates a string of the minimum cost warp path
     * 
     * Distances array must be populated through use of distance method
     * 
     * @return Path
     */
    public String printMinCostWarpPath(){
        return findPath(this.distances.length-1, this.distances[0].length-1);
    }
    
    /**
     * Recursive method that finds and prints the minimum warped path
     * 
     * @param int i position in distances, should be max of series
     * @param int j position in distances, should be max of series
     * 
     * @return current position
     */
    protected String findPath(int i, int j){

        double prevDistance = this.distances[i][j];
        int oldI = i;
        int oldJ = j;
        
        //final condition
        if(i != 0 || j != 0){
            //decrementing i and j
            if(i > 0 && j > 0){
                double min = Math.min(this.distances[i-1][j], Math.min(this.distances[i-1][j-1], this.distances[i][j-1]));
                if(this.distances[i-1][j-1] == min){
                    i--;
                    j--;
                }else if(this.distances[i-1][j] == min){
                    i--;
                }else if(this.distances[i][j-1] == min){
                    j--;
                }
            }else if(j > 0){
                j--;
            }else if(i > 0){
                i--;
            }

            //recursive step
            return "("+oldI+","+oldJ+") = "+prevDistance+"\n" + findPath(i,j);
        }else{
            return "("+oldI+","+oldJ+") = "+prevDistance+"\n";
        }
    }
    
    
    /**
     * returns the Euclidean distances array
     * 
     * @return double[][] distances
     */
    public double[][] getDistanceArray(){
        return this.distances;
    }
    
    /**
     * This will print the diagonal route with no warping
     */
    public void printDiagonalRoute(){
        System.out.println("------------------ Diagonal Route ------------------");
        for(int i = this.distances.length-1; i >= 0; i--){
            System.out.print(this.distances[i][i]+" ");
        }
        System.out.println("\n------------------ End ------------------");
    }
    
    /**
     * Prints the distances array as a table
     */
    public void printDistances(){        
        System.out.println("------------------ Distances Table ------------------");
        for(int i = 0; i<this.distances.length; i++){
            System.out.print("Row ="+i+" = ");
            for(int j = 0; j<this.distances[0].length; j++){
                System.out.print(" "+ distances[i][j]);
            }
            System.out.print("\n");
        }
        System.out.println("------------------ End ------------------");
    }

    /**
     * Check if early abandon enabled
     * 
     * @return early abandon enabled
     */
    public boolean isEarlyAbandon() {
        return isEarlyAbandon;
    }

    /**
     * Set early abandon
     * 
     * @param isEarlyAbandon value for early abandon
     */
    public void setIsEarlyAbandon(boolean isEarlyAbandon) {
        this.isEarlyAbandon = isEarlyAbandon;
    }

    @Override
    public String toString() {
        return "BasicDTW{ " + "earlyAbandon=" + this.isEarlyAbandon + " }";
    }
}