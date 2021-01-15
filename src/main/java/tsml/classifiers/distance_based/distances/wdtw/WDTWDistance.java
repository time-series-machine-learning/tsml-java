package tsml.classifiers.distance_based.distances.wdtw;

import tsml.classifiers.distance_based.distances.MatrixBasedDistanceMeasure;
import tsml.classifiers.distance_based.distances.twed.TWEDistance;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.data_containers.TimeSeriesInstance;

import java.util.Arrays;

import static tsml.classifiers.distance_based.distances.dtw.DTWDistance.cost;

/**
 * WDTW distance measure.
 * <p>
 * Contributors: goastler
 */
public class WDTWDistance
    extends MatrixBasedDistanceMeasure implements WDTW {

    private double g = 0.05;
    private double[] weights = new double[0];

    @Override
    public double getG() {
        return g;
    }

    @Override
    public void setG(double g) {
        this.g = g;
    }
    
    private void generateWeights(int length) {
        if(weights.length < length) {
            final double halfLength = (double) length / 2;
            double[] oldWeights = weights;
            weights = new double[length];
            System.arraycopy(oldWeights, 0, weights, 0, oldWeights.length);
            for(int i = oldWeights.length; i < length; i++) {
                weights[i] = 1d / (1d + Math.exp(-g * (i - halfLength)));
            }
        }
    }

    @Override
    public double distance(final TimeSeriesInstance a, final TimeSeriesInstance b, final double limit) {
        final int aLength = a.getMaxLength();
        final int bLength = b.getMaxLength();
        setup(aLength, bLength, true);
        final double lengthRatio = (double) bLength / aLength;
        final double windowSize = 1d * bLength;
        
        // start and end of window
        int start = 0;
        int j = start;
        int i = 0;
        double mid;
        int end =  (int) Math.min(bLength - 1, Math.ceil(windowSize));
        int prevEnd;
        double[] row = getRow(i);
        double[] prevRow;
        
        // generate weights
        generateWeights(Math.max(aLength, bLength));
        
        // process top left cell of mat
        double min = row[j] = weights[j] * cost(a, i, b, j);
        j++;
        // compute the first row
        for(; j <= end; j++) {
            row[j] = row[j - 1] + weights[j] * cost(a, i, b, j);
            min = Math.min(min, row[j]);
        }
        if(min > limit) return Double.POSITIVE_INFINITY; // quit if beyond limit
        i++;
//        System.out.println(Arrays.toString(row));
        
        // process remaining rows
        for(; i < aLength; i++) {
            // reset min for the row
            min = Double.POSITIVE_INFINITY;
            
            // start, end and mid of window
            prevEnd = end;
            mid = i * lengthRatio;
            start = (int) Math.max(0, Math.floor(mid - windowSize));
            end = (int) Math.min(bLength - 1, Math.ceil(mid + windowSize));
            j = start;
            
            // change rows
            prevRow = row;
            row = getRow(i);
            
            // set the top values outside of window to inf
            Arrays.fill(prevRow, prevEnd + 1, end + 1, Double.POSITIVE_INFINITY);
            // set the value left of the window to inf
            if(j > 0) row[j - 1] = Double.POSITIVE_INFINITY;
            
            // if assessing the left most column then only mapping option is top - not left or topleft
            if(j == 0) {
                row[j] = prevRow[j] + weights[Math.abs(i - j)] * cost(a, i, b, j);
                min = Math.min(min, row[j++]);
            }
            // compute the distance for each cell in the row
            for(; j <= end; j++) {
                row[j] = Math.min(prevRow[j], Math.min(row[j - 1], prevRow[j - 1])) + weights[Math.abs(i - j)] * cost(a, i, b, j);;
                min = Math.min(min, row[j]);
            }
            if(min > limit) return Double.POSITIVE_INFINITY; // quit if beyond limit

//            System.out.println(Arrays.toString(row));
        }
        
        // last value in the current row is the distance
        final double distance = row[row.length - 1];
        teardown();
        return distance;
    }

    @Override
    public ParamSet getParams() {
        return super.getParams().add(WDTW.G_FLAG, g);
    }

    @Override
    public void setParams(final ParamSet param) throws Exception {
        super.setParams(param);
        ParamHandlerUtils.setParam(param, WDTW.G_FLAG, this::setG, Double::parseDouble);
    }

    public static void main(String[] args) {

        final WDTWDistance dm = new WDTWDistance();
        dm.setG(0.93);
        dm.setGenerateDistanceMatrix(true);
        System.out.println(dm.distance(new TimeSeriesInstance(new double[][]{{-0.61669188, -0.61457225, -0.614337, -0.61229702, -0.60490284, -0.60015683, -0.60251477, -0.59914029, -0.5994704, -0.59890877, -0.60149498, -0.59845271, -0.59720598, -0.59805173, -0.59496309, -0.59781352, -0.59791231, -0.60049775, -0.60346294, -0.60432485, -0.6059935, -0.61060773, -0.61241943, -0.6142363, -0.6144902, -0.61527331, -0.61503777, -0.61356555, -0.6150875, -0.61578875, -0.61611581, -0.61466587, -0.61700124, -0.6152232, -0.61102496, -0.61213302, -0.61255943, -0.60887692, -0.60648618, -0.60195352, -0.60050856, -0.59361796, -0.57397131, -0.53338711, -0.44440891, -0.31054872, -0.085413503, 0.16140997, 0.37696739, 0.59358564, 0.74143177, 0.84372506, 0.94500956, 1.0539945, 1.1874815, 1.3489619, 1.5163414, 1.6641137, 1.7748155, 1.8604024, 1.8632145, 1.8783673, 1.8845416, 1.8936771, 1.8856951, 1.8863657, 1.8888039, 1.892227, 1.8843643, 1.8868293, 1.8917485, 1.8903223, 1.8914252, 1.8875782, 1.8892389, 1.8867303, 1.8867303, 1.8783243, 1.8414894, 1.8077406, 1.7536198, 1.6705217, 1.5517774, 1.4222274, 1.2689417, 1.1358604, 1.0139831, 0.8967505, 0.78706847, 0.68423735, 0.58046103, 0.50441083, 0.23339878, 0.092860954, -0.079262216, -0.19633795, -0.31372726, -0.45303586, -0.55902926, -0.64103997, -0.6856257, -0.70596701, -0.70677689, -0.69576724, -0.68623515, -0.68337171, -0.68106149, -0.67715521, -0.67032877, -0.66498154, -0.65976848, -0.65480616, -0.65260428, -0.65091106, -0.64961145, -0.64917806, -0.64814383, -0.65056603, -0.66204341, -0.67084259, -0.68024777, -0.69212403, -0.70130726, -0.70887941, -0.7112296, -0.71867218, -0.72617892, -0.73071827, -0.7397023, -0.74451592, -0.7496083, -0.75397475, -0.75540088, -0.7570875, -0.75702027, -0.75956861, -0.76041092, -0.76007488, -0.75753925, -0.7599693, -0.7578577, -0.75986679, -0.75790934, -0.75720589, -0.75268949, -0.74701004, -0.73868853, -0.73022874, -0.72150435, -0.71683532}}), new TimeSeriesInstance(new double[][] {{-1.162563, -1.1583252, -1.1536569, -1.1513451, -1.1448405, -1.1418631, -1.1393674, -1.1383064, -1.1365658, -1.1333474, -1.1326546, -1.1281729, -1.1299554, -1.1118926, -1.0584936, -0.97912769, -0.90292665, -0.83160777, -0.78092644, -0.76410888, -0.74825676, -0.75434836, -0.76044297, -0.78157212, -0.7878509, -0.79612498, -0.79694981, -0.79416557, -0.7768767, -0.75854807, -0.74387109, -0.72902831, -0.71671207, -0.70168144, -0.67661632, -0.62555606, -0.544198, -0.42737632, -0.28135866, -0.11188097, 0.07787528, 0.24691163, 0.4036547, 0.55144659, 0.66712022, 0.79256739, 0.87210921, 0.9447132, 1.031816, 1.1276312, 1.1427345, 1.1688782, 1.1874118, 1.1975574, 1.2061246, 1.2059004, 1.2325493, 1.2429288, 1.2129546, 1.2173248, 1.2216273, 1.2259664, 1.2283049, 1.2376053, 1.2191452, 1.2413216, 1.2427624, 1.2463334, 1.2613342, 1.2696288, 1.2629024, 1.2428215, 1.2420639, 1.2407128, 1.2244077, 1.2198746, 1.2198746, 1.2246345, 1.2368664, 1.2402425, 1.2319588, 1.2348571, 1.2463733, 1.2321702, 1.2451417, 1.2497568, 1.2522344, 1.2715283, 1.2402908, 1.2316748, 1.2358025, 1.2433159, 1.2594393, 1.2402202, 1.250318, 1.2118949, 1.1892697, 1.1780458, 1.1182586, 1.01455, 0.91969086, 0.8118989, 0.66131786, 0.52045488, 0.35872114, 0.20310731, 0.03112537, -0.12838055, -0.27340235, -0.3988132, -0.52341348, -0.62551194, -0.68108477, -0.72337102, -0.76702493, -0.78627176, -0.79017399, -0.791183, -0.79094879, -0.7867044, -0.77730912, -0.78277633, -0.78734863, -0.79069632, -0.79522861, -0.80466145, -0.81499455, -0.82008193, -0.8259019, -0.82349603, -0.83268485, -0.85094373, -0.8594099, -0.88670496, -0.90954036, -0.94463497, -1.0015843, -1.05519, -1.1141787, -1.142881, -1.1462166, -1.144096, -1.1403992, -1.1355837, -1.1149366, -1.1039109, -1.0874788, -1.0756754, -1.0671332, -1.0666325}})));
    }
}
