
package Bags2DShapelets;

import java.util.Objects;
import weka.core.Instance;
import static utilities.multivariate_tools.MultivariateInstanceTools.splitMultivariateInstance;
import weka.core.Instances;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class Shapelet2D implements Comparable<Shapelet2D> {

    public double [][] shapelet = null;
    ShapeletSummary summary;
    public int size = -1; 
    public boolean normalised = false;
    
    public double score = -1.0;
    
    public static class ShapeletSummary {
        public Instance originatingImg = null;
        public int xStart = -1;
        public int yStart = -1;
        public int xLen = -1;
        public int yLen = -1;

        public ShapeletSummary(Instance originatingImg, int xStart, int yStart, int xLen, int yLen) {
            this.originatingImg = originatingImg;
            this.xStart = xStart;
            this.yStart = yStart;
            this.xLen = xLen;
            this.yLen = yLen;
        }
        
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            
            if (o == null || getClass() != o.getClass()) return false;        
            ShapeletSummary other = (ShapeletSummary) o;
            
            if (xStart != other.xStart) return false;
            if (yStart != other.yStart) return false;
            if (xLen != other.xLen) return false;
            if (yLen != other.yLen) return false;
            
            return !originatingImg.equals(other.originatingImg);
            
            
        }

        @Override
        public int hashCode() {
            int hash = 5;
            hash = 59 * hash + Objects.hashCode(this.originatingImg);
            hash = 59 * hash + this.xStart;
            hash = 59 * hash + this.yStart;
            hash = 59 * hash + this.xLen;
            hash = 59 * hash + this.yLen;
            return hash;
        }
    }
    
    public Shapelet2D() {
        
    }
    
    public Shapelet2D(Instance img, int xStart, int yStart, int xLen, int yLen) {
        summary = new ShapeletSummary(img, xStart, yStart, xLen, yLen);
        extract(img, xStart, yStart, xLen, yLen);
        normalise();
    }
    
    public Shapelet2D(ShapeletSummary summary) {
        extract(summary.originatingImg, summary.xStart, summary.yStart, summary.xLen, summary.yLen);
        normalise();
    }
    
    public void extract(Instance img, int xStart, int yStart, int xLen, int yLen) {
        if (yLen < 1) yLen = xLen; //assume square if yStart not specified
        
        this.shapelet = new double[xLen][yLen];
        
        Instance[] instsRows = splitMultivariateInstance(img);
        
        for (int sx = xStart, tx = 0; sx < xStart+xLen; sx++, tx++)
            for (int sy = yStart, ty = 0; sy < yStart+yLen; sy++, ty++)
                this.shapelet[tx][ty] = instsRows[sx].value(sy);
        
        this.size = xLen * yLen;
    }
    
    /**
     * @return exact distance
     */
    public double distanceTo(Shapelet2D other) {
        double dist = .0;
        
        for (int x = 0; x < summary.xLen; x++)
            for (int y = 0; y < summary.yLen; y++)
                dist += (other.shapelet[x][y] - this.shapelet[x][y]) * (other.shapelet[x][y] - this.shapelet[x][y]);
        
        return dist;
    }
    
    /**
     * @param bestDistSoFar test for early abandon
     * @return exact distance if less than bestDistSoFar, otherwise _some_ value greater than bestDistSoFar
     */
    public double distanceTo_EarlyAbandon(Shapelet2D other, double bestDistSoFar) {
        double dist = .0;
        boolean earlyAbandon = false;
        
        for (int x = 0; x < summary.xLen && !earlyAbandon; x++) {
            for (int y = 0; y < summary.yLen; y++) {
                dist += (other.shapelet[x][y] - this.shapelet[x][y]) * (other.shapelet[x][y] - this.shapelet[x][y]);

                if (dist > bestDistSoFar) {
                    earlyAbandon = true;
                    break;
                }
            }
        }
        
        return dist;
    }
            
            
    @Override
    public int compareTo(Shapelet2D other) {
        int c = Double.compare(this.score, other.score);
        if (c == 0)
            return (-1) * Integer.compare(this.size, other.size);
        else 
            return c;
    }
    
    public boolean isBetterThan(Shapelet2D other) { 
        return this.compareTo(other) > 0;
    }
    
    /**
     * For now, simply scaling to range 0-1, same as the overall images are themselves 
     */
    public void normalise() {
        if (normalised)
            return; 
                   
        double min = Double.MAX_VALUE;
        double max = -Double.MAX_VALUE;
        
        for (int i = 0; i < shapelet.length; i++) {
            for (int j = 0; j < shapelet[i].length; j++) {
                if (shapelet[i][j] < min)
                    min = shapelet[i][j];
                if (shapelet[i][j] > max)
                    max = shapelet[i][j];
            }
        }
        
        for (int i = 0; i < shapelet.length; i++)
            for (int j = 0; j < shapelet[i].length; j++)
                shapelet[i][j] = (shapelet[i][j] - min) / (max-min);
        
        normalised = true;
    }
}
