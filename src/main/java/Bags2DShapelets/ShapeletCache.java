
package Bags2DShapelets;

import Bags2DShapelets.Shapelet2D.ShapeletSummary;
import java.util.HashMap;
import java.util.Map;

/**
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class ShapeletCache {

    
    public static Map<ShapeletSummary,Shapelet2D> cache = new HashMap<>(20000);
    
    /**
     * will use shapelet's own summary object as key
     */
    public static void put(Shapelet2D shapelet) {
        cache.put(shapelet.summary, shapelet);
    }
    
    public static void put(ShapeletSummary key, Shapelet2D shapelet) {
        cache.put(key, shapelet);
    }
    
    public static Shapelet2D getAndPutIfNotExists(ShapeletSummary key) {
        Shapelet2D result = cache.get(key);
    
        if (result == null) {
            result = new Shapelet2D(key);
            cache.put(key, result);
        } 
    
        return result;
    }
    
    public static Shapelet2D get(ShapeletSummary key) {
        return cache.get(key);
    }
    
    
}
