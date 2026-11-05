# spec/peft_checkpoint_spec.cr

require "./peft_spec_helper"

describe SHAInet::PEFT::Checkpoint do
  it "round-trips adapter tensors and configuration exactly" do
    path = File.tempname("peft", ".json")
    begin
      saved = PEFTSpec::ScaleAdapter.new(3.5_f32)
      SHAInet::PEFT::Checkpoint.save(path, saved)
      loaded = PEFTSpec::ScaleAdapter.new
      config = SHAInet::PEFT::Checkpoint.load_into(path, loaded)
      loaded.parameters["scale"].data.should eq([3.5_f32])
      config["value"].as_f.should be_close(3.5, 1e-6)
      File.exists?("#{path}.#{Process.pid}.tmp").should be_false
    ensure
      File.delete?(path)
    end
  end

  it "returns kind, config, and tensors without an adapter" do
    path = File.tempname("peft", ".json")
    begin
      SHAInet::PEFT::Checkpoint.save(path, PEFTSpec::ScaleAdapter.new(1.25_f32))
      kind, config, tensors = SHAInet::PEFT::Checkpoint.load(path)
      kind.should eq("scale")
      config["value"].as_f.should be_close(1.25, 1e-6)
      tensors["scale"].data.should eq([1.25_f32])
    ensure
      File.delete?(path)
    end
  end

  it "rejects kind mismatches" do
    path = File.tempname("peft", ".json")
    begin
      SHAInet::PEFT::Checkpoint.save(path, PEFTSpec::ScaleAdapter.new)
      adapter = PEFTSpec::ScaleAdapter.new
      payload = File.read(path).gsub("\"scale\"", "\"lora\"")
      payload = payload.sub("\"kind\":\"lora\"", "\"kind\":\"other\"")
      File.write(path, payload)
      expect_raises(ArgumentError, "kind") do
        SHAInet::PEFT::Checkpoint.load_into(path, adapter)
      end
    ensure
      File.delete?(path)
    end
  end

  it "rejects unsupported versions and non-finite tensors" do
    path = File.tempname("peft", ".json")
    begin
      SHAInet::PEFT::Checkpoint.save(path, PEFTSpec::ScaleAdapter.new)
      versioned = File.read(path).sub("\"version\":1", "\"version\":99")
      File.write(path, versioned)
      expect_raises(ArgumentError, "version") do
        SHAInet::PEFT::Checkpoint.load(path)
      end
      infinite = File.read(path).sub("\"version\":99", "\"version\":1").sub("2.0", "Infinity")
      File.write(path, infinite)
      expect_raises(Exception) do
        SHAInet::PEFT::Checkpoint.load(path)
      end
    ensure
      File.delete?(path)
    end
  end

  it "rejects non-finite parameters on save" do
    path = File.tempname("peft", ".json")
    begin
      adapter = PEFTSpec::ScaleAdapter.new
      adapter.parameters["scale"][0, 0] = Float32::INFINITY
      expect_raises(ArgumentError, "finite") do
        SHAInet::PEFT::Checkpoint.save(path, adapter)
      end
      File.exists?(path).should be_false
    ensure
      File.delete?(path)
    end
  end
end
