<template>
    <div class="all">
        <data-table ref="dataTable"/>
    </div>
</template>

<script>
    import DataTable from "./DataTable";
    export default {
        name: "DataImport",
        components: {DataTable},
    }
</script>

<style scoped>
    .all {
        text-align: left;
        font-family: "Helvetica Neue", Helvetica, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "微软雅黑", Arial, sans-serif;
        font-size: 15px;
        color: #606266;
    }
</style>