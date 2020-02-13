package tsml.classifiers.distance_based.knn;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

public class KnnTest {

    @Before
    public void setUp() throws Exception {
    }

    @After
    public void tearDown() throws Exception {
    }

    @Test
    public void setSavePath() {
        Knn knn = new Knn();
        boolean result = knn.setSavePath("some/path/to/a/folder");
        assertFalse(result);
    }

    @Test
    public void unsetSavePath() {
        Knn knn = new Knn();
        boolean result = knn.setSavePath(null);
        assertFalse(result);
    }

    @Test
    public void getSavePath() {
        Knn knn = new Knn();
        String path = "some/path/to/a/folder";
        knn.setSavePath(path);
        String savePath = knn.getSavePath();
        assertEquals(savePath, path);
    }

    @Test
    public void getSavePathDefaultNull() {
        Knn knn = new Knn();
        String savePath = knn.getSavePath();
        assertNull(savePath);
    }

    @Test
    public void isIgnorePreviousCheckpoints() {
    }

    @Test
    public void setIgnorePreviousCheckpoints() {
    }

    @Test
    public void setMinCheckpointIntervalNanos() {
    }

    @Test
    public void getMinCheckpointIntervalNanos() {
    }

    @Test
    public void getMemoryWatcher() {
    }

    @Test
    public void isRandomTieBreak() {
    }

    @Test
    public void setRandomTieBreak() {
    }

    @Test
    public void isEarlyAbandon() {
    }

    @Test
    public void setEarlyAbandon() {
    }

    @Test
    public void getK() {
    }

    @Test
    public void setK() {
    }

    @Test
    public void getDistanceFunction() {
    }

    @Test
    public void setDistanceFunction() {
    }
}