package utilities;

import org.junit.Assert;
import org.junit.Test;

import java.io.File;

public class FileUtilsTest {
    
    @Test
    public void testFileLock() {
        final File file = new File("hello.txt");
        final FileUtils.FileLock lock = new FileUtils.FileLock(file);
        Assert.assertEquals(file.getPath() + ".lock", lock.getLockFile().getPath());
        Assert.assertTrue(lock.getLockFile().exists());
        Assert.assertEquals(1, lock.getLockCount());
        lock.unlock();
        Assert.assertFalse(lock.getLockFile().exists());
        Assert.assertEquals(0, lock.getLockCount());
        lock.lock();
        Assert.assertTrue(lock.getLockFile().exists());
        Assert.assertEquals(1, lock.getLockCount());
        lock.lock();
        Assert.assertTrue(lock.getLockFile().exists());
        Assert.assertEquals(2, lock.getLockCount());
        lock.unlock();
        Assert.assertTrue(lock.getLockFile().exists());
        Assert.assertEquals(1, lock.getLockCount());
        lock.unlock();
        Assert.assertFalse(lock.getLockFile().exists());
        Assert.assertEquals(0, lock.getLockCount());
    }
}
