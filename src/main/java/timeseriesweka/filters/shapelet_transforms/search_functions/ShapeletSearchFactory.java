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
package timeseriesweka.filters.shapelet_transforms.search_functions;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;


/**
 * 
 *
 * @author Aaron
 */
public class ShapeletSearchFactory {
    private static final List<Function<ShapeletSearchOptions, ShapeletSearch>> searchConstructors = createSearchConstructors();
    //{FULL, FS, GENETIC, RANDOM, LOCAL, MAGNIFY, TIMED_RANDOM, SKIPPING, TABU, REFINED_RANDOM, IMP_RANDOM, SUBSAMPLE, SKEWED};

    ShapeletSearchOptions options;
    
    public ShapeletSearchFactory(ShapeletSearchOptions ops){
        options = ops;
    }
    
    private static List<Function<ShapeletSearchOptions, ShapeletSearch>> createSearchConstructors(){
        List<Function<ShapeletSearchOptions, ShapeletSearch>> sCons = new ArrayList();
        sCons.add(ShapeletSearch::new);
        sCons.add(FastShapeletSearch::new);
        sCons.add(GeneticSearch::new);
        sCons.add(RandomSearch::new);
        sCons.add(LocalSearch::new);
        sCons.add(MagnifySearch::new);
        sCons.add(RandomTimedSearch::new);
        sCons.add(SkippingSearch::new);
        sCons.add(TabuSearch::new);
        sCons.add(RefinedRandomSearch::new);
        sCons.add(ImpRandomSearch::new);
        sCons.add(SubsampleRandomSearch::new);
        sCons.add(SkewedRandomSearch::new);
        return sCons;
    }
    
    public ShapeletSearch getShapeletSearch(){
        return searchConstructors.get(options.getSearchType().ordinal()).apply(options);
    }
    
    
    public static void main(String[] args) {
        System.out.println(new ShapeletSearchFactory(new ShapeletSearchOptions.Builder()
                                                    .setSearchType(ShapeletSearch.SearchType.FULL)
                                                    .build())
                                                    .getShapeletSearch());
    }
}
