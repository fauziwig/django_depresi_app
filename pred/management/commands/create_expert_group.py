from django.core.management.base import BaseCommand
from django.contrib.auth.models import Group, User


class Command(BaseCommand):
    help = 'Create Expert group and manage expert users'

    def add_arguments(self, parser):
        parser.add_argument(
            '--create-group',
            action='store_true',
            help='Create the Expert group',
        )
        parser.add_argument(
            '--add-user',
            type=str,
            help='Add a user to the Expert group by username',
        )
        parser.add_argument(
            '--remove-user',
            type=str,
            help='Remove a user from the Expert group by username',
        )
        parser.add_argument(
            '--list-experts',
            action='store_true',
            help='List all users in the Expert group',
        )

    def handle(self, *args, **options):
        if options['create_group']:
            self.create_expert_group()
        
        if options['add_user']:
            self.add_user_to_expert_group(options['add_user'])
        
        if options['remove_user']:
            self.remove_user_from_expert_group(options['remove_user'])
        
        if options['list_experts']:
            self.list_expert_users()

    def create_expert_group(self):
        """Create the Expert group if it doesn't exist"""
        group, created = Group.objects.get_or_create(name='Expert')
        if created:
            self.stdout.write(
                self.style.SUCCESS('Successfully created Expert group')
            )
        else:
            self.stdout.write(
                self.style.WARNING('Expert group already exists')
            )

    def add_user_to_expert_group(self, username):
        """Add a user to the Expert group"""
        try:
            user = User.objects.get(username=username)
            expert_group, created = Group.objects.get_or_create(name='Expert')
            
            if expert_group in user.groups.all():
                self.stdout.write(
                    self.style.WARNING(f'User {username} is already an Expert')
                )
            else:
                user.groups.add(expert_group)
                self.stdout.write(
                    self.style.SUCCESS(f'Successfully added {username} to Expert group')
                )
        except User.DoesNotExist:
            self.stdout.write(
                self.style.ERROR(f'User {username} does not exist')
            )

    def remove_user_from_expert_group(self, username):
        """Remove a user from the Expert group"""
        try:
            user = User.objects.get(username=username)
            expert_group = Group.objects.get(name='Expert')
            
            if expert_group in user.groups.all():
                user.groups.remove(expert_group)
                self.stdout.write(
                    self.style.SUCCESS(f'Successfully removed {username} from Expert group')
                )
            else:
                self.stdout.write(
                    self.style.WARNING(f'User {username} is not an Expert')
                )
        except User.DoesNotExist:
            self.stdout.write(
                self.style.ERROR(f'User {username} does not exist')
            )
        except Group.DoesNotExist:
            self.stdout.write(
                self.style.ERROR('Expert group does not exist')
            )

    def list_expert_users(self):
        """List all users in the Expert group"""
        try:
            expert_group = Group.objects.get(name='Expert')
            experts = expert_group.user_set.all()
            
            if experts:
                self.stdout.write(self.style.SUCCESS('Expert users:'))
                for expert in experts:
                    full_name = expert.get_full_name() or 'No full name'
                    self.stdout.write(f'  - {expert.username} ({full_name}) - {expert.email}')
            else:
                self.stdout.write(self.style.WARNING('No users in Expert group'))
        except Group.DoesNotExist:
            self.stdout.write(
                self.style.ERROR('Expert group does not exist. Run --create-group first.')
            )
