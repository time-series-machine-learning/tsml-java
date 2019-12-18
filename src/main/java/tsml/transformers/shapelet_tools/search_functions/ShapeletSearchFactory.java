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
package tsml.transformers.shapelet_tools.search_functions;


/**
 * 
 *
 * @author Aaron
 */
public class ShapeletSearchFactory {
    ShapeletSearchOptions options;
    
    public ShapeletSearchFactory(ShapeletSearchOptions ops){
        options = ops;
    }
  /**
      * @return a Shapelet search configured by the options
     */
    public ShapeletSearch getShapeletSearch(){
        switch(options.getSearchType()){
            case FULL:
                return new ShapeletSearch(options);
            case RANDOM:
                return new RandomSearch(options);
            default:
                throw new UnsupportedOperationException(" Currently only FULL and RANDOM shapelet search are allowed" +
                        "you passed" + options.getSearchType()+" the others are in package aaron_search and are not debugged");
        }
    }

//    private static final List<Function<ShapeletSearchOptions, ShapeletSearch>> searchConstructors = createSearchConstructors();
    //{FULL, FS, GENETIC, RANDOM, LOCAL, MAGNIFY, TIMED_RANDOM, SKIPPING, TABU, REFINED_RANDOM, IMP_RANDOM, SUBSAMPLE, SKEWED};
/*
    //Aaron likes C++. This is just an indexed list of constructors for possible search technique
    private static List<Function<ShapeletSearchOptions, ShapeletSearch>> createSearchConstructors(){
        List<Function<ShapeletSearchOptions, ShapeletSearch>> sCons = new ArrayList();
        sCons.add(ShapeletSearch::new);
        sCons.add(RandomSearch::new);
//      All the below have been moved to aaron_search. The constructors are all protected for some reason
//so that needs refactoring to be used here.
        sCons.add(BayesianOptimisedSearch::new);
        sCons.add(FastShapeletSearch::new);
        sCons.add(GeneticSearch::new);
        sCons.add(LocalSearch::new);
        sCons.add(MagnifySearch::new);
        sCons.add(RandomTimedSearch::new);
        sCons.add(SkippingSearch::new);
        sCons.add(TabuSearch::new);
        sCons.add(RefinedRandomSearch::new);
        sCons.add(ImprovedRandomSearch::new);
        sCons.add(SubsampleRandomSearch::new);
        sCons.add(SkewedRandomSearch::new);
        return sCons;
    }

*/

    public static void main(String[] args) {
        System.out.println(new ShapeletSearchFactory(new ShapeletSearchOptions.Builder()
                                                    .setSearchType(ShapeletSearch.SearchType.FULL)
                                                    .build())
                                                    .getShapeletSearch());
    }
}
