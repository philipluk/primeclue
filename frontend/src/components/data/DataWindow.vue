<template>
    <div>
        <top-menu active="data-window"/>
        <data-list/>
    </div>
</template>

<script>
    import DataList from "./DataList";
    export default {
        name: "DataWindow",
        components: {DataList}
    }
</script>

<style scoped>

</style>