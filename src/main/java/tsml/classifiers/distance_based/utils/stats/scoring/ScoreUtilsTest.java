package tsml.classifiers.distance_based.utils.stats.scoring;

import org.junit.Assert;
import org.junit.Test;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class ScoreUtilsTest {

    /////////////////////////////////////////////////////////////////////// info entropy / info score testing

    @Test
    public void testInfoEntropyImpure() {
        double v = ScoreUtils.infoEntropy(new int[] {5, 5});
        Assert.assertEquals(1, v, 0.0);
    }

    @Test
    public void testInfoEntropyManyClassImpure() {
        double v = ScoreUtils.infoEntropy(new int[] {5, 5, 5, 5, 5, 5, 5, 5});
        Assert.assertEquals(3, v, 0.0);
    }

    @Test
    public void testInfoScoreImpure() {
        double v = ScoreUtils.infoScore(new int[] {5, 5});
        Assert.assertEquals(0, v, 0.0);
    }

    @Test
    public void testInfoScoreManyClassImpure() {
        double v = ScoreUtils.infoScore(new int[] {5, 5, 5, 5, 5, 5, 5, 5});
        Assert.assertEquals(0, v, 0.0);
    }
    
    @Test
    public void testInfoEntropyPure() {
        double v = ScoreUtils.infoEntropy(new int[] {10, 0});
        Assert.assertEquals(0, v, 0.0);
    }

    @Test
    public void testInfoEntropyPureManyClass() {
        double v = ScoreUtils.infoEntropy(new int[] {10, 0, 0, 0, 0, 0, 0, 0});
        Assert.assertEquals(0, v, 0.0);
    }

    @Test
    public void testInfoScorePure() {
        double v = ScoreUtils.infoScore(new int[] {10, 0});
        Assert.assertEquals(1, v, 0.0);
    }

    @Test
    public void testInfoScorePureManyClass() {
        double v = ScoreUtils.infoScore(new int[] {10, 0, 0, 0, 0, 0, 0, 0});
        Assert.assertEquals(3, v, 0.0);
    }

    @Test
    public void testInfoEntropyA() {
        double v = ScoreUtils.infoEntropy(new int[] {4, 6});
        Assert.assertEquals(0.9709505944546686, v, 0.0);
    }

    @Test
    public void testInfoEntropyB() {
        double v = ScoreUtils.infoEntropy(new int[] {1, 9});
        Assert.assertEquals(0.4689955935892812, v, 0.0);
    }

    //////////////////////////////////////////////////////////////////////////// gini score / gini entropy testing

    @Test
    public void testGiniImpurityImpure() {
        double v = ScoreUtils.giniImpurity(new int[] {5, 5});
        Assert.assertEquals(0.5, v, 0.0);
    }

    @Test
    public void testGiniImpurityManyClassImpure() {
        double v = ScoreUtils.giniImpurity(new int[] {5, 5, 5, 5, 5, 5, 5, 5});
        Assert.assertEquals(0.875, v, 0.0);
    }

    @Test
    public void testGiniScoreImpure() {
        double v = ScoreUtils.giniScore(new int[] {5, 5});
        Assert.assertEquals(0, v, 0.0);
    }

    @Test
    public void testGiniScoreManyClassImpure() {
        double v = ScoreUtils.giniScore(new int[] {5, 5, 5, 5, 5, 5, 5, 5});
        Assert.assertEquals(0, v, 0.0);
    }

    @Test
    public void testGiniImpurityPure() {
        double v = ScoreUtils.giniImpurity(new int[] {10, 0});
        Assert.assertEquals(0, v, 0.0);
    }

    @Test
    public void testGiniScorePure() {
        double v = ScoreUtils.giniScore(new int[] {10, 0});
        Assert.assertEquals(1, v, 0.0);
    }

    @Test
    public void testGiniImpurityPureManyClass() {
        double v = ScoreUtils.giniImpurity(new int[] {10, 0, 0, 0, 0, 0, 0, 0});
        Assert.assertEquals(0, v, 0.0);
    }

    @Test
    public void testGiniScorePureManyClass() {
        double v = ScoreUtils.giniScore(new int[] {10, 0, 0, 0, 0, 0, 0, 0});
        Assert.assertEquals(1.75, v, 0.0);
    }

    @Test
    public void testGiniImpurityA() {
        double v = ScoreUtils.giniImpurity(new int[] {4, 6});
        Assert.assertEquals(0.48, v, 0.0);
    }

    @Test
    public void testGiniImpurityB() {
        double v = ScoreUtils.giniImpurity(new int[] {1, 9});
        Assert.assertEquals(0.17999999999999994, v, 0.0);
    }
    
    //////////////////////////////////////////////////////////////////////////////// gini gain testing


    @Test
    public void testGiniGainPure() {
        double v = ScoreUtils.infoGain(new int[] {8, 4},
                new int[][] {
                        {8, 0},
                        {0, 4},
                });
        Assert.assertEquals(0.9182958340544896, v, 0.0);
    }

    @Test
    public void testGiniGainImpure() {
        double v = ScoreUtils.infoGain(new int[] {8, 4},
                new int[][] {
                        {4, 2},
                        {2, 4},
                });
        Assert.assertEquals(0, v, 0.0);
    }

    @Test
    public void testGiniGainA() {
        double v = ScoreUtils.giniGain(new int[] {8, 4},
                new int[][] {
                        {7, 3},
                        {1, 1},
                });
        Assert.assertEquals(0.011111111111111127, v, 0.0);
    }

    @Test
    public void testGiniGainB() {
        double v = ScoreUtils.giniGain(new int[] {8, 4},
                new int[][] {
                        {7, 1},
                        {1, 3},
                });
        Assert.assertEquals(0.17361111111111116, v, 0.0);
    }

    /////////////////////////////////////////////////////////////////////////////// info gain testing

    @Test
    public void testInfoGainPure() {
        double v = ScoreUtils.infoGain(new int[] {8, 4},
                new int[][] {
                        {8, 0},
                        {0, 4},
                });
        Assert.assertEquals(0.9182958340544896, v, 0.0);
    }

    @Test
    public void testInfoGainImpure() {
        double v = ScoreUtils.infoGain(new int[] {8, 4},
                new int[][] {
                        {4, 2},
                        {2, 4},
                });
        Assert.assertEquals(0, v, 0.0);
    }

    @Test
    public void testInfoGainA() {
        double v = ScoreUtils.infoGain(new int[] {8, 4},
                new int[][] {
                        {7, 3},
                        {1, 1},
                });
        Assert.assertEquals(v, 0.01722008469557898, 0.0000000000000001);
    }

    @Test
    public void testInfoGainB() {
        double v = ScoreUtils.infoGain(new int[] {8, 4},
                new int[][] {
                        {7, 1},
                        {1, 3},
                });
        Assert.assertEquals(0.28549349710171434, v, 0.0);
    }

    //////////////////////////////////////////////////////////////////// test chi square

    @Test
    public void testChiSquaredImpure() {
        double v = ScoreUtils.chiSquared(new int[] {8, 4},
                new int[][] {
                        {4, 2},
                        {4, 2},
                });
        Assert.assertEquals(0, v, 0.0);
    }

    @Test
    public void testChiSquaredPure() {
        double v = ScoreUtils.chiSquared(new int[] {8, 4},
                new int[][] {
                        {8, 0},
                        {0, 4},
                });
        Assert.assertEquals(12, v, 0.0);
    }

    @Test
    public void testChiSquaredA() {
        double v = ScoreUtils.chiSquared(new int[] {8, 4},
                new int[][] {
                        {7, 1},
                        {1, 3},
                });
        Assert.assertEquals(4.6875, v, 0.0);
    }

    @Test
    public void testChiSquaredB() {
        double v = ScoreUtils.chiSquared(new int[] {8, 4},
                new int[][] {
                        {7, 3},
                        {1, 1},
                });
        Assert.assertEquals(0.30000000000000004, v, 0.0);
    }
}
