<template>
    <div>
        <div style="text-align: left; font-size: large">
            Classifying for classifier: {{this.$props.classifierName}}
        </div>
        <div style="text-align: left; margin-top: 15px">
            <div style="display: inline-flex">
                <el-upload :auto-upload="false" action="" :on-change="fileSelected" :show-file-list="false">
                    <el-button style="width: 150px;" size="small" type="primary">Load file</el-button>
                </el-upload>
            </div>
        </div>
        <div>
            <classification-data-table ref="dataTable" :classifierName="this.classifier"/>
        </div>
    </div>
</template>

<script>
    import ClassificationDataTable from "./ClassificationDataTable";
    export default {
        name: "Classification",
        props: ['classifierName'],
        components: {ClassificationDataTable},
        data() {
            return {
                classifier: this.$props.classifierName,
            }
        },

        methods: {
            fileSelected(file) {
                const reader = new FileReader();
                reader.onload = () => {
                    this.$refs.dataTable.redrawTable(reader.result);
                };
                reader.readAsText(file.raw);
            }
        }
    }
</script>

<style scoped>

</style>