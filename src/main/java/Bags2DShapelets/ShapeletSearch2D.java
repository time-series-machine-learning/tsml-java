
package Bags2DShapelets;

import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Random;
import timeseriesweka.filters.shapelet_transforms.OrderLineObj;
import timeseriesweka.filters.shapelet_transforms.class_value.NormalClassValue;
import timeseriesweka.filters.shapelet_transforms.quality_measures.InformationGain;
import timeseriesweka.filters.shapelet_transforms.quality_measures.ShapeletQuality;
import timeseriesweka.filters.shapelet_transforms.quality_measures.ShapeletQualityMeasure;
import utilities.class_distributions.ClassDistribution;
import static utilities.multivariate_tools.MultivariateInstanceTools.channelLength;
import static utilities.multivariate_tools.MultivariateInstanceTools.convertMultiInstanceToArrays;
import static utilities.multivariate_tools.MultivariateInstanceTools.splitMultivariateInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class ShapeletSearch2D {
    
    Random rng = null;
    
    public boolean squaresOnly = true; //if false, x and y dimensions in the random shapelet search may be different
    
    public int sDistStride = -1; //will default to max(1, minImageDimensionLength/100) if not set
    public int minShapeletLength = -1; //will default to 3 if not set
    public int maxShapeletLength = -1; //will default to minImageDimensionLength if not set
    public int k = -1; //will default to num instances (images) if not set 
    public int numShapeletsToSearch = 1000; //will generate 1000 candidates by default, top k taken in fnial transform
    
    ClassDistribution classDistribution = null;
    ShapeletQualityMeasure shapeletEvaluator = (new ShapeletQuality(ShapeletQuality.ShapeletQualityChoice.INFORMATION_GAIN)).getQualityMeasure(); //y u do dis aaron...
    
    public ShapeletSearch2D(int seed) {
        rng = new Random(seed);
    }
    
    
    public Shapelet2D[] generateKShapelets(Instances imgs) {
        //init search hyperparas to defaults if not already set
        if (k < 1) k = imgs.numInstances();
        if (minShapeletLength < 1) minShapeletLength = 3;
        if (maxShapeletLength < 1) maxShapeletLength = channelLength(imgs);
        if (sDistStride < 1) sDistStride = Math.max(1, maxShapeletLength / 100);
        
        
        //calc/store class distribution for shapelet quality measure later
        NormalClassValue classVal = new NormalClassValue();
        classVal.init(imgs);
        classDistribution = classVal.getClassDistributions();
        
        //init internal search paras
        int shapeletsPerTimeSeries = numShapeletsToSearch / imgs.numInstances();
        int extraShapeletsToFind = numShapeletsToSearch % imgs.numInstances();
        
        //the head will the the shapelet with the worst score (and biggest size, in case of ties)
        PriorityQueue<Shapelet2D> bestShapeletsSoFar = new PriorityQueue<>();
        
        //find as many shapelets per image as needed
        for (Instance img : imgs) {
            for (int i = 0; i < shapeletsPerTimeSeries; i++) {
                Shapelet2D candidate = generateCandidateFromSingleImage(img);
                candidate.score = evaluateShapelet(imgs, candidate);
                addIfBetter(bestShapeletsSoFar, candidate, k);
            }
        }
        
        //if any left to find, get them from random images 
        for (int i = 0; i < extraShapeletsToFind; i++) {
            Shapelet2D candidate = generateCandidateFromAnyImage(imgs);
            candidate.score = evaluateShapelet(imgs, candidate);
            addIfBetter(bestShapeletsSoFar, candidate, k);
        }
        
        //convert to array and return the best
        return bestShapeletsSoFar.toArray(new Shapelet2D[] { });
    }
    
    public double[][] imgInstTo2dArray(Instance img) {
        return convertMultiInstanceToArrays(splitMultivariateInstance(img));
    }
    
    public double sDist2D(Instance imgInst, Shapelet2D candidate) {
        double[][] img = imgInstTo2dArray(imgInst);
        
        //todo double check whether the class value is still included in this, don't know how class values 
        //are stored for multivariate instances, e.g is it duplicated for each channel? if so, is still here
        //if not, it's already beed removed
        
        double bestDist = Double.MAX_VALUE;
        
        int xLen = candidate.summary.xLen;
        int yLen = candidate.summary.yLen;
        
        for (int xStart = 0; xStart < img.length-xLen; xStart += sDistStride) {
            for (int yStart = 0; yStart < img[0].length-yLen; yStart += sDistStride) {
                
                Shapelet2D.ShapeletSummary summary = new Shapelet2D.ShapeletSummary(imgInst, xStart, yStart, xLen, yLen);
//                Shapelet2D region = ShapeletCache.getAndPutIfNotExists(summary);
                Shapelet2D region = new Shapelet2D(summary);

                double dist = candidate.distanceTo_EarlyAbandon(region, bestDist);
                
                if (dist < bestDist) 
                    bestDist = dist;
                
            }
        }
        
        return bestDist;
    }
    
    //todo i presume i should remove the instance that the shapelet originally came from,
    //distance would just be 0. would also need to update the calss distribution? 
    public double evaluateShapelet(Instances imgs, Shapelet2D candidate) {
        double score = -1.0;
        
        List<OrderLineObj> orderLine = new ArrayList<>(imgs.numInstances());
        
        for (Instance img : imgs) {
            double distance = sDist2D(img, candidate);
            orderLine.add(new OrderLineObj(distance, img.classValue()));
        }
        
        return shapeletEvaluator.calculateQuality(orderLine, classDistribution);
    }
    
    //todo check for self similarity here too 
    private void addIfBetter(PriorityQueue<Shapelet2D> bestShapeletsSoFar, Shapelet2D candidate, int k) {
        if (bestShapeletsSoFar.size() < k)
            bestShapeletsSoFar.add(candidate);
        else 
            if (candidate.isBetterThan(bestShapeletsSoFar.peek())) { //better than the worst of the best so far
                bestShapeletsSoFar.poll();
                bestShapeletsSoFar.add(candidate);
            }
    }
    
    public Shapelet2D generateCandidateFromAnyImage(Instances imgs) {
        return generateCandidateFromSingleImage(imgs.get(rng.nextInt(imgs.numInstances())));
    }
    
    public Shapelet2D generateCandidateFromSingleImage(Instance img) {
        Instance[] instRows = splitMultivariateInstance(img);
        int imgX = instRows.length;
        int imgY = instRows[0].numAttributes();
        
        
        int maxStart = maxShapeletLength - minShapeletLength;
        int xStart = rng.nextInt(maxStart);
        int yStart = rng.nextInt(maxStart);
        
        int xLen = -1;
        int yLen = -1;
        
        int maxXlen = Math.min(maxShapeletLength, imgX-xStart);
        int maxYlen = Math.min(maxShapeletLength, imgY-yStart);
        
        if (squaresOnly) {
            int maxLen = Math.min(maxXlen, maxYlen);
            
            //want value in range min to max inclusive
            //suppose max = 5, min = 2
            //rng.nextInt((5+1)) gives value in range 0 to 5 
            //rng.nextInt((5+1)-2)+2 gives value in range 2 to 5
            xLen = rng.nextInt(maxLen - minShapeletLength) + minShapeletLength;
            yLen = xLen;
        } else {
            xLen = rng.nextInt(maxXlen - minShapeletLength) + minShapeletLength;
            yLen = rng.nextInt(maxYlen - minShapeletLength) + minShapeletLength;
        }
        
        return new Shapelet2D(img, xStart, yStart, xLen, yLen);
    }
    
//    public int evaluateCandidate(Shapelet2D) {
//        
//    }
    
}
