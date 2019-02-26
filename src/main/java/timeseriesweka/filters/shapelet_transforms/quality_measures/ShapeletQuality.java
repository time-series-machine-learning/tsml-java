/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package timeseriesweka.filters.shapelet_transforms.quality_measures;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.Supplier;
import timeseriesweka.filters.shapelet_transforms.OrderLineObj;
import utilities.class_counts.ClassCounts;


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
    private static final List<BiFunction<ClassCounts, Integer, ShapeletQualityBound>>  boundConstructor = createBound();
    private static List<Supplier<ShapeletQualityMeasure>> createQuality(){
        List<Supplier<ShapeletQualityMeasure>> cons = new ArrayList<>();
        cons.add(InformationGain::new);
        cons.add(FStat::new);
        cons.add(KruskalWallis::new);
        cons.add(MoodsMedian::new);
        return cons;
    }
    
    private static List<BiFunction<ClassCounts, Integer, ShapeletQualityBound>> createBound(){
        List<BiFunction<ClassCounts, Integer, ShapeletQualityBound>> cons = new ArrayList();
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
    
    public void initQualityBound(ClassCounts classDist, int percentage){
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
