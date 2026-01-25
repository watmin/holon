    # Compare
    if time_batch > 0:
        speedup = time_ind / time_batch
        print("\n🎯 Results:")
        print(".3f")
        print(".3f")
        print(".2f")
        if speedup > 1.5:
            print("✅ Bulk insert significantly faster!")
        else:
            print("📈 Bulk insert still more efficient for larger batches")
