package utilities.params;

import scala.annotation.meta.param;
import tsml.classifiers.distance_based.distances.*;
import utilities.ArrayUtilities;

import java.util.*;

/**
 * holds a mapping of parameter names to their corresponding values, where the values are stored as a ParamValues
 * object to allow for sub parameter spaces.
 */
public class Params {

    /**
     * holds a set of values (e.g. DTW and DDTW) and a set of corresponding params for those values (e.g. a set of
     * warping windows). I.e. many values can map to many sub spaces.
     */
    public static class ParamValues {
        private List<Object> values = new ArrayList<>();
        private List<Params> params = new ArrayList<>();

        public int[] getBins() {
            int[] bins = new int[params.size() + 1];
            for(int i = 0; i < params.size(); i++) {
                bins[i] = params.get(i).size();
            }
            bins[bins.length - 1] = values.size();
            return bins;
        }

        public int size() {
            return ArrayUtilities.numPermutations(getBins());
        }

        public Param.ParamValue get(final int index) {
            int[] indices = ArrayUtilities.fromPermutation(index, getBins());
            Param.ParamValue paramValue = new Param.ParamValue();
            for(int i = 0; i < params.size(); i++) {
                Param param = params.get(i).get(indices[i]);
                paramValue.addParam(param);
            }
            Object value = values.get(indices[indices.length - 1]);
            paramValue.setValue(value);
            return paramValue;
        }

        public ParamValues() {}

        public ParamValues(List<Object> values, List<Params> params) {
            setValues(values);
            setParams(params);
        }

        public ParamValues(List<Object> values) {
            this(values, null); // no sub param space
        }

        public void addValues(Object... values) {
            this.values.addAll(Arrays.asList(values));
        }

        public void addParams(Params... params) {
            this.params.addAll(Arrays.asList(params));
        }

        public List<Object> getValues() {
            return values;
        }

        public void setValues(List<Object> values) {
            if(values == null) {
                values = new ArrayList<>();
            }
            this.values = values;
        }

        public List<Params> getParams() {
            return params;
        }

        public void setParams(List<Params> params) {
            if(params == null) {
                params = new ArrayList<>();
            }
            this.params = params;
        }

        @Override public String toString() {
            return "ParamValues{" +
                "values=" + values +
                ", params=" + params +
                '}';
        }
    }

    public int[] getBins() {
        int[] bins = new int[paramsMap.size()];
        Iterator<Map.Entry<String, List<ParamValues>>> iterator = paramsMap.entrySet().iterator();
        for(int i = 0; i < bins.length; i++) {
            Map.Entry<String, List<ParamValues>> entry = iterator.next();
            int size = 0;
            for(ParamValues paramValues : entry.getValue()) {
                size += paramValues.size();
            }
            bins[i] = size;
        }
        return bins;
    }

    public Param get(int index) {
        int[] indices = ArrayUtilities.fromPermutation(index, getBins());
        int i = 0;
        Param param = new Param();
        for(Map.Entry<String, List<ParamValues>> entry : paramsMap.entrySet()) {
            index = indices[i];
            List<ParamValues> paramValuesList = entry.getValue();
            for(ParamValues paramValues : paramValuesList) {
                int size = paramValues.size();
                index -= size;
                if(index < 0) {
                    index += size;
                    Param.ParamValue paramValue = paramValues.get(index);
                    param.add(entry.getKey(), paramValue);
                    break;
                }
            }
            i++;
        }
        return param;
//        for(Map.Entry<String, List<ParamValues>> entry : paramsMap.entrySet()) {
//            List<ParamValues> paramValuesList = entry.getValue();
//            for(ParamValues paramValues : paramValuesList) {
//                int size = paramValues.size();
//                index -= size;
//                if(index < 0) {
//                    index += size;
//                    Param.ParamValue paramValue = paramValues.get(index);
//                    Param param = new Param();
//                    param.add(entry.getKey(), paramValue);
//                    return param;
//                }
//            }
//        }
//        throw new IndexOutOfBoundsException();
    }

    public int size() {
//        int size = 1;
//        for(Map.Entry<String, List<ParamValues>> entry : paramsMap.entrySet()) {
//            int subSize = 0;
//            List<ParamValues> paramValuesList = entry.getValue();
//            for(ParamValues paramValues : paramValuesList) {
//                subSize += paramValues.size();
//            }
//            size *= subSize;
//        }
//        return size;
        return ArrayUtilities.numPermutations(getBins());
    }

    private Map<String, List<ParamValues>> paramsMap = new LinkedHashMap<>(); // 1-many mapping of parameter names

    public void add(String name, ParamValues param) {
        paramsMap.computeIfAbsent(name, k -> new ArrayList<>()).add(param);
    }

    @Override public String toString() {
        return "Params{" +
            "paramsMap=" + paramsMap +
            '}';
    }

    public static void main(String[] args) {
        Params params = new Params();
        Params wParams = new Params();
        wParams.add(Dtw.WARPING_WINDOW_FLAG, new ParamValues(Arrays.asList(1,2,3,4,5)));
        params.add(DistanceMeasure.DISTANCE_FUNCTION_FLAG, new ParamValues(Arrays.asList(new Dtw(), new Ddtw()),
                                                                           Arrays.asList(wParams)));
        Params lParams = new Params();
        lParams.add(Wdtw.G_FLAG, new ParamValues(Arrays.asList(1, 2, 3)));
        lParams.add(Lcss.EPSILON_FLAG, new ParamValues(Arrays.asList(1, 2, 3, 4)));
        params.add(DistanceMeasure.DISTANCE_FUNCTION_FLAG, new ParamValues(Arrays.asList(new Wdtw(), new Wddtw()),
                                                                           Arrays.asList(lParams)));
        int size;
        size = wParams.size();
        size = lParams.size();
        size = params.size();
        for(int i = 0; i < size; i++) {
            System.out.println(i);
            Param param = params.get(i);
            System.out.println(param);
        }
    }


}
