#include <iostream>
#include <memory>
#include <vector>
#include <fstream>

#include "kompute/Kompute.hpp"

using namespace std;

typedef std::vector<std::vector<float> > Matrix;

// Forward declaring our helper function to compile the shader source into spirv
static std::vector<uint32_t> compileSource( const std::string& source);

std::vector<float> flatten(const std::vector<std::vector<float>> &orig)
{   
    std::vector<float> ret;
    for(const auto &v: orig)
        ret.insert(ret.end(), v.begin(), v.end());                                                                                         
    return ret;
}   


vector<string> SplitString(string s){
	vector<string> v;
	string temp = "";
	for(int i=0;i<s.length();++i){
		
		if(s[i]==' '){
			v.push_back(temp);
			temp = "";
		}
		else{
			temp.push_back(s[i]);
		}
		
	}
	v.push_back(temp);
	
    return v;
}

void print(Matrix& vec){
    for(const auto& i: vec){
        for(const auto& j: i){
            cout << j << ' ';
        }
        cout << endl;
    }
}

void write(const string& name, Matrix& vec){
    ofstream out(name);
    for(const auto& i: vec){
        for(const auto& j: i){
            out << j << ' ';
        }
        out << '\n';
    }
}

int main()
{
    Matrix K;
    Matrix img;
    string str;
    vector<string> vec;
    vector<float> in;
    ifstream infile1("/home/jesse/Desktop/kompute_test/Kompute/K.txt");
    while(getline(infile1, str)){
        vec = SplitString(str);
        for(const auto& s: vec){
            in.push_back(stof(s));
        }
        K.push_back(in);
        in.clear();
    }

    ifstream infile2("/home/jesse/Desktop/kompute_test/Kompute/img.txt");
    while(getline(infile2, str)){
        vec = SplitString(str);
        for(const auto& s: vec){
            in.push_back(stof(s));
        }
        img.push_back(in);
        in.clear();
    }

    kp::Manager mgr(0);

    vector<float> k_flat = flatten(K);

    vector<float> img_flat = flatten(img);

    vector<float> result(img_flat.size(), 0);

    auto tensorInA = mgr.tensor(img_flat);

    auto tensorInB = mgr.tensor(k_flat);

    auto tensorOut = mgr.tensor(result);

    auto tensorSizes = mgr.tensor({ (float)img.size(), (float)img[0].size(), (float)K.size()});


    std::string shader(R"(
        // The version to use 
        #version 450
        //#extension GL_EXT_debug_printf : enable
        //#extension  VK_KHR_portability_subset : enable

        // The buffers are provided via the tensors
        layout(binding = 0) buffer bufA { float image[]; };
        layout(binding = 1) buffer bufB { float kernel[]; };
        layout(binding = 2) buffer bufOut { float result[]; };
        layout(binding = 3) buffer bufSizes { float sizes[]; };
    

        layout (local_size_x = 1, local_size_y = 1) in;
        void main() {
            ivec2 ourPos = ivec2(gl_GlobalInvocationID.xy);

            int kernel_width = int(sizes[2]);
            int kernel_height = int(sizes[2]);

            int ii;
            int jj;

            int ki;
            int kj;

            float temp = 0;

            for (int i=-int(floor(kernel_width/2.0)); i<floor(kernel_width/2.0); i++){
                for (int j=-int(floor(kernel_height/2.0)); j<floor(kernel_height/2.0); j++){
                    
                    ki = int(floor(kernel_width/2.0) + i);
                    kj = int(floor(kernel_height/2.0) + j);
 
                    ii = ourPos.x%int(sizes[1]) + i;
                    jj = ourPos.x/int(sizes[1]) + j;

                    //if(row + (kernel_offset - iFlip) >= 0 && row + (kernel_offset - iFlip) < row_size && col + (kernel_offset - jFlip) >= 0 && col + (kernel_offset - jFlip) < col_size) {
                    if (ii >= 0 && ii < int(sizes[0]) && jj >= 0 && jj < int(sizes[1])) {
                        //temp += image[row + (kernel_offset - iFlip) * col_size + col + (kernel_offset - jFlip)] * kernel[iFlip * kernel_size + jFlip];
                        temp += image[ii * int(sizes[1]) + jj] * kernel[ki*int(sizes[2]) + kj];
                    }
                    
                }
            }

            //debugPrintfEXT("temp is: %f", temp);

            result[ourPos.x%int(sizes[1])*int(sizes[1]) + ourPos.x/int(sizes[1])] = temp;

        }


      )");
    std::cout << "main" << std::endl;
    std::vector<std::shared_ptr<kp::Tensor>> params = { tensorInA, tensorInB, tensorOut, tensorSizes };

    std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(params, compileSource(shader));

    mgr.sequence()
        ->record<kp::OpTensorSyncDevice>(params)
        ->record<kp::OpAlgoDispatch>(algo)
        ->record<kp::OpTensorSyncLocal>(params)
        ->eval();

    // prints "Output {  0  4  12  }"
    std::cout<< "Output: {  ";
    for (const float& elem : tensorOut->vector()) {
      std::cout << elem << "  ";
    }
    std::cout << "}" << std::endl;
}

static std::vector<uint32_t> compileSource( const std::string& source)
{
    std::ofstream fileOut("tmp_kp_shader.comp");
    fileOut << source;
    fileOut.close();
    if (system(std::string("glslangValidator -V tmp_kp_shader.comp -o tmp_kp_shader.comp.spv").c_str()))
        throw std::runtime_error("Error running glslangValidator command");
    std::ifstream fileStream("tmp_kp_shader.comp.spv", std::ios::binary);
    std::vector<char> buffer;
    buffer.insert(buffer.begin(), std::istreambuf_iterator<char>(fileStream), {});
    return {(uint32_t*)buffer.data(), (uint32_t*)(buffer.data() + buffer.size())};
}

