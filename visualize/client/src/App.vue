<template>
    <div id="app">
        <el-container>
            <el-header>Header</el-header>
            <el-main>
                <el-row>
                    <el-col :span="18">
                        <el-input
                                placeholder="请输入句子"
                                v-model="sentence"
                                clearable>
                        </el-input>
                    </el-col>
                    <el-col :span="6">
                        <el-button v-on:click="getSegment">确定</el-button>
                    </el-col>
                </el-row>
                <el-row>
                    <el-col :span="18">
                        <el-input
                                placeholder="分词结果"
                                v-model="segment"
                                :disabled="true">
                        </el-input>
                    </el-col>
                </el-row>
            </el-main>
        </el-container>
    </div>
</template>
<style>
    .el-header {
        background-color: #B3C0D1;
        color: #333;
        text-align: center;
        line-height: 60px;
    }

    .el-main {
        background-color: #E9EEF3;
        color: #333;
        text-align: center;
        line-height: 100px;
    }

    body > .el-container {
        margin-bottom: 40px;
    }
</style>
<script>
    import axios from 'axios'

    export default {
        data() {
            return {
                sentence: '',
                segment: ''
            }
        },
        methods: {
            getSegment() {
                axios.post('http://127.0.0.1:5000/predict',
                    {sentence: this.sentence})
                    .then((response) => {
                        let data = response.data;
                        this.segment = data.segment.join(',')
                    }).catch((error) => {
                        console.error(error)
                })
            }
        }
    }
</script>
