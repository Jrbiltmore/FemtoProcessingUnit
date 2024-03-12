# fpu/src/cloud/Scalable_Resource_Management.py

import boto3
import googleapiclient.discovery
from azure.mgmt.compute import ComputeManagementClient
from azure.identity import DefaultAzureCredential

class AWSEC2Manager:
    def __init__(self, region_name='us-east-1'):
        self.ec2 = boto3.resource('ec2', region_name=region_name)

    def create_instance(self, image_id, instance_type, key_name, min_count=1, max_count=1):
        instances = self.ec2.create_instances(
            ImageId=image_id,
            InstanceType=instance_type,
            KeyName=key_name,
            MinCount=min_count,
            MaxCount=max_count
        )
        return instances

    def list_instances(self):
        instances = self.ec2.instances.filter(
            Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]
        )
        for instance in instances:
            print(instance.id, instance.instance_type)

class GCEManager:
    def __init__(self, project, zone):
        self.compute = googleapiclient.discovery.build('compute', 'v1')
        self.project = project
        self.zone = zone

    def create_instance(self, name, machine_type, image_project, image_family):
        image_response = self.compute.images().getFromFamily(
            project=image_project, family=image_family).execute()
        source_disk_image = image_response['selfLink']
        
        machine_type = f"zones/{self.zone}/machineTypes/{machine_type}"
        
        config = {
            'name': name,
            'machineType': machine_type,
            'disks': [
                {
                    'boot': True,
                    'autoDelete': True,
                    'initializeParams': {
                        'sourceImage': source_disk_image,
                    }
                }
            ],
            'networkInterfaces': [{
                'network': 'global/networks/default',
                'accessConfigs': [
                    {'type': 'ONE_TO_ONE_NAT', 'name': 'External NAT'}
                ]
            }],
        }

        return self.compute.instances().insert(
            project=self.project,
            zone=self.zone,
            body=config).execute()

class AzureVMManager:
    def __init__(self, subscription_id):
        self.subscription_id = subscription_id
        self.compute_client = ComputeManagementClient(credential=DefaultAzureCredential(), subscription_id=subscription_id)

    def create_vm(self, resource_group, vm_name, location, vm_size, image_reference):
        vm_parameters = {
            'location': location,
            'hardware_profile': {
                'vm_size': vm_size
            },
            'storage_profile': {
                'image_reference': {
                    'id': image_reference
                }
            },
            'os_profile': {
                'computer_name': vm_name,
                'admin_username': 'yourusername',
                'admin_password': 'yourpassword'
            },
            'network_profile': {
                'network_interfaces': [{
                    'id': f'/subscriptions/{self.subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Network/networkInterfaces/{vm_name}-nic',
                }]
            }
        }

        creation_result = self.compute_client.virtual_machines.begin_create_or_update(
            resource_group_name=resource_group,
            vm_name=vm_name,
            parameters=vm_parameters
        )
        return creation_result.result()

# Example usage
if __name__ == "__main__":
    # AWS EC2 Example
    aws_manager = AWSEC2Manager()
    aws_manager.create_instance(image_id='ami-0c02fb55956c7d316', instance_type='t2.micro', key_name='your-key-name')

    # Google Cloud Engine Example
    gce_manager = GCEManager(project='your-project-id', zone='us-central1-a')
    gce_manager.create_instance(name='test-instance', machine_type='f1-micro', image_project='debian-cloud', image_family='debian-10')

    # Azure VM Example
    azure_manager = AzureVMManager(subscription_id='your-subscription-id')
    azure_manager.create_vm(resource_group='your-resource-group', vm_name='test-vm', location='eastus', vm_size='Standard_DS1_v2', image_reference='/subscriptions/your-subscription-id/resourceGroups/your-resource-group/providers/Microsoft.Compute/images/your-custom-image')
