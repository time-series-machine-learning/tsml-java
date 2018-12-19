/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.filters.shapelet_transforms.quality_measures;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.Supplier;
import timeseriesweka.filters.shapelet_transforms.OrderLineObj;
import utilities.class_distributions.ClassDistribution;


/**
 *
 * @author raj09hxu
 */


// a container class for a ShapeletQualityMeasure and an optional bounding class.
public class ShapeletQuality {
    
    public enum ShapeletQualityChoice
    {

        /**
         * Used to specify that the filter will use Information Gain as the
         * shapelet quality measure (introduced in Ye & Keogh 2009)
         */
        INFORMATION_GAIN,
        /**
         * Used to specify that the filter will use F-Stat as the shapelet
         * quality measure (introduced in Lines et. al 2012)
         */
        F_STAT,
        /**
         * Used to specify that the filter will use Kruskal-Wallis as the
         * shapelet quality measure (introduced in Lines and Bagnall 2012)
         */
        KRUSKALL_WALLIS,
        /**
         * Used to specify that the filter will use Mood's Median as the
         * shapelet quality measure (introduced in Lines and Bagnall 2012)
         */
        MOODS_MEDIAN
    }

    public ShapeletQualityChoice getChoice() {
        return choice;
    }

    public ShapeletQualityMeasure getQualityMeasure() {
        return qualityMeasure;
    }

    public Optional<ShapeletQualityBound> getBound() {
        return bound;
    }
    
    ShapeletQualityChoice choice;
    ShapeletQualityMeasure qualityMeasure;
    Optional<ShapeletQualityBound> bound = Optional.empty();
    
    //init static lists of constructors.
    private static final List<Supplier<ShapeletQualityMeasure>> qualityConstructors = createQuality();
    private static final List<BiFunction<ClassDistribution, Integer, ShapeletQualityBound>>  boundConstructor = createBound();
    private static List<Supplier<ShapeletQualityMeasure>> createQuality(){
        List<Supplier<ShapeletQualityMeasure>> cons = new ArrayList<>();
        cons.add(InformationGain::new);
        cons.add(FStat::new);
        cons.add(KruskalWallis::new);
        cons.add(MoodsMedian::new);
        return cons;
    }
    
    private static List<BiFunction<ClassDistribution, Integer, ShapeletQualityBound>> createBound(){
        List<BiFunction<ClassDistribution, Integer, ShapeletQualityBound>> cons = new ArrayList();
        cons.add(InformationGainBound::new);
        cons.add(FStatBound::new);
        cons.add(KruskalWallisBound::new);
        cons.add(MoodsMedianBound::new);
        return cons;
    }
    
    public ShapeletQuality(ShapeletQualityChoice choice){
        this.choice = choice;
        qualityMeasure = qualityConstructors.get(choice.ordinal()).get();
    }
    
    public void initQualityBound(ClassDistribution classDist, int percentage){
        bound = Optional.of(boundConstructor.get(choice.ordinal()).apply(classDist, percentage));
    }
    
    public void setBsfQuality(double bsf){
        if(bound.isPresent())
            bound.get().setBsfQuality(bsf);
    }
    
    public boolean pruneCandidate(){
        return bound.isPresent() && bound.get().pruneCandidate();
    }
    
    public void updateOrderLine(OrderLineObj obj){
        if(bound.isPresent())
            bound.get().updateOrderLine(obj);
    }
    
}
